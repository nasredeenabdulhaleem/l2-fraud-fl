// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

/// @title FLAggregator
/// @notice L2-native coordination and audit layer for federated fraud-detection training.
/// @dev Design principle: on an optimistic rollup, persistent storage writes remain the
///      dominant gas cost even though execution and calldata are cheap relative to L1.
///      A model gradient is far too large to store as state, so this contract stores ONLY
///      fixed-size 32-byte commitments to off-chain update payloads (content hashes) plus
///      small counters. Heavy aggregation and Byzantine-robust filtering happen off chain
///      in the Flower server; the contract records which submissions were admitted so the
///      aggregation remains tamper-evident and auditable. Every write is bounded and
///      fixed-size, which keeps per-transaction gas predictable and unit-testable.
contract FLAggregator {
    // ---------------------------------------------------------------------
    // Roles
    // ---------------------------------------------------------------------

    /// @notice The federation coordinator (the entity running the Flower server).
    address public immutable COORDINATOR;

    /// @notice Registered client nodes permitted to submit updates.
    mapping(address => bool) public isClient;

    /// @notice Number of registered clients (kept as a counter to avoid iterating a set).
    uint64 public clientCount;

    // ---------------------------------------------------------------------
    // Round lifecycle
    // ---------------------------------------------------------------------

    struct Round {
        bytes32 baseModelRef;    // commitment to the global model distributed at round start
        bytes32 globalModelRef;  // commitment to the aggregated global model at finalisation
        uint64 deadline;         // unix timestamp after which submissions are rejected
        uint32 submissionCount;  // number of client updates recorded this round
        bool open;               // true between openRound and finaliseRound
        bool finalised;          // true after finaliseRound
    }

    /// @notice Monotonic identifier of the current round (0 means no round has opened yet).
    uint64 public currentRoundId;

    /// @notice roundId => Round metadata.
    mapping(uint64 => Round) public rounds;

    /// @notice roundId => client => committed update reference (bytes32(0) means no submission).
    mapping(uint64 => mapping(address => bytes32)) public updateRef;

    // ---------------------------------------------------------------------
    // Events (off-chain telemetry subscribes to these)
    // ---------------------------------------------------------------------

    event ClientRegistered(address indexed client);
    event ClientRemoved(address indexed client);
    event RoundOpened(uint64 indexed roundId, bytes32 baseModelRef, uint64 deadline);
    event UpdateSubmitted(uint64 indexed roundId, address indexed client, bytes32 updateRef);
    event RoundFinalised(uint64 indexed roundId, bytes32 globalModelRef, uint32 submissionCount);

    // ---------------------------------------------------------------------
    // Errors (custom errors are cheaper than require strings)
    // ---------------------------------------------------------------------

    error NotCoordinator();
    error NotClient();
    error RoundNotOpen();
    error RoundStillOpen();
    error AlreadyFinalised();
    error DeadlinePassed();
    error AlreadySubmitted();
    error EmptyRef();
    error AlreadyRegistered();
    error UnknownClient();

    // ---------------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------------

    modifier onlyCoordinator() {
        _onlyCoordinator();
        _;
    }

    function _onlyCoordinator() internal view {
        if (msg.sender != COORDINATOR) revert NotCoordinator();
    }

    modifier onlyClient() {
        _onlyClient();
        _;
    }

    function _onlyClient() internal view {
        if (!isClient[msg.sender]) revert NotClient();
    }

    constructor() {
        COORDINATOR = msg.sender;
    }

    // ---------------------------------------------------------------------
    // Client registry
    // ---------------------------------------------------------------------

    /// @notice Register a client node permitted to submit updates.
    function registerClient(address client) external onlyCoordinator {
        if (isClient[client]) revert AlreadyRegistered();
        isClient[client] = true;
        unchecked { clientCount += 1; }
        emit ClientRegistered(client);
    }

    /// @notice Remove a client (for example, one flagged as Byzantine off chain).
    function removeClient(address client) external onlyCoordinator {
        if (!isClient[client]) revert UnknownClient();
        isClient[client] = false;
        unchecked { clientCount -= 1; }
        emit ClientRemoved(client);
    }

    // ---------------------------------------------------------------------
    // Round lifecycle
    // ---------------------------------------------------------------------

    /// @notice Open a new federated round.
    /// @param baseModelRef 32-byte commitment to the global model distributed to clients.
    /// @param submissionWindow seconds clients have to submit before the deadline.
    function openRound(bytes32 baseModelRef, uint64 submissionWindow)
        external
        onlyCoordinator
        returns (uint64 roundId)
    {
        if (baseModelRef == bytes32(0)) revert EmptyRef();
        // Guard: cannot open a new round while the previous one is still open.
        if (currentRoundId != 0 && rounds[currentRoundId].open) revert RoundStillOpen();

        unchecked { currentRoundId += 1; }
        roundId = currentRoundId;

        uint64 deadline = uint64(block.timestamp) + submissionWindow;
        rounds[roundId] = Round({
            baseModelRef: baseModelRef,
            globalModelRef: bytes32(0),
            deadline: deadline,
            submissionCount: 0,
            open: true,
            finalised: false
        });

        emit RoundOpened(roundId, baseModelRef, deadline);
    }

    /// @notice Submit a fixed-size commitment to this client's local update.
    /// @param ref 32-byte content hash of the off-chain update payload.
    /// @dev Hot path. Bounded to a single SSTORE of a non-zero word plus a counter bump.
    function submitUpdate(bytes32 ref) external onlyClient {
        uint64 roundId = currentRoundId;
        Round storage r = rounds[roundId];

        if (!r.open) revert RoundNotOpen();
        if (block.timestamp > r.deadline) revert DeadlinePassed();
        if (ref == bytes32(0)) revert EmptyRef();
        if (updateRef[roundId][msg.sender] != bytes32(0)) revert AlreadySubmitted();

        updateRef[roundId][msg.sender] = ref;
        unchecked { r.submissionCount += 1; }

        emit UpdateSubmitted(roundId, msg.sender, ref);
    }

    /// @notice Finalise the current round with a commitment to the aggregated global model.
    /// @param globalModelRef 32-byte commitment to the newly aggregated global model.
    /// @dev The coordinator may finalise once the round has enough submissions or the
    ///      deadline has passed. Off-chain aggregation determines the globalModelRef.
    function finaliseRound(bytes32 globalModelRef) external onlyCoordinator {
        uint64 roundId = currentRoundId;
        Round storage r = rounds[roundId];

        if (!r.open) revert RoundNotOpen();
        if (r.finalised) revert AlreadyFinalised();
        if (globalModelRef == bytes32(0)) revert EmptyRef();

        r.open = false;
        r.finalised = true;
        r.globalModelRef = globalModelRef;

        emit RoundFinalised(roundId, globalModelRef, r.submissionCount);
    }

    // ---------------------------------------------------------------------
    // Views
    // ---------------------------------------------------------------------

    /// @notice Read the committed update reference for a client in a given round.
    function getUpdateRef(uint64 roundId, address client) external view returns (bytes32) {
        return updateRef[roundId][client];
    }

    /// @notice Read full round metadata.
    function getRound(uint64 roundId) external view returns (Round memory) {
        return rounds[roundId];
    }
}
