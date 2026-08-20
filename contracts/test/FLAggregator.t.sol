// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

import {Test} from "forge-std/Test.sol";
import {FLAggregator} from "../src/FLAggregator.sol";

/// @notice Foundry test suite. Asserts functional correctness, permission edge cases,
///         and explicit gas bounds on the two hot paths so a gas regression fails the build.
contract FLAggregatorTest is Test {
    FLAggregator internal agg;

    address internal coordinator = address(this);
    address internal clientA = address(0xA11CE);
    address internal clientB = address(0xB0B);
    address internal stranger = address(0xBAD);

    bytes32 internal constant BASE_REF = keccak256("base-model-v0");
    bytes32 internal constant UPDATE_A = keccak256("update-a");
    bytes32 internal constant UPDATE_B = keccak256("update-b");
    bytes32 internal constant GLOBAL_REF = keccak256("global-model-v1");

    // Gas ceilings for the hot paths. Tune to the observed profile plus a safety margin.
    uint256 internal constant MAX_GAS_SUBMIT = 80_000;
    uint256 internal constant MAX_GAS_FINALISE = 60_000;

    function setUp() public {
        agg = new FLAggregator();
        agg.registerClient(clientA);
        agg.registerClient(clientB);
    }

    // ------------------------------------------------------------------
    // Happy path
    // ------------------------------------------------------------------

    function test_FullRoundLifecycle() public {
        uint64 roundId = agg.openRound(BASE_REF, 1 days);
        assertEq(roundId, 1);

        vm.prank(clientA);
        agg.submitUpdate(UPDATE_A);
        vm.prank(clientB);
        agg.submitUpdate(UPDATE_B);

        FLAggregator.Round memory r = agg.getRound(roundId);
        assertEq(r.submissionCount, 2);
        assertTrue(r.open);

        agg.finaliseRound(GLOBAL_REF);

        r = agg.getRound(roundId);
        assertTrue(r.finalised);
        assertEq(r.globalModelRef, GLOBAL_REF);
        assertEq(agg.getUpdateRef(roundId, clientA), UPDATE_A);
    }

    function test_ClientRegistryCount() public {
        assertEq(agg.clientCount(), 2);
        agg.removeClient(clientB);
        assertEq(agg.clientCount(), 1);
        assertFalse(agg.isClient(clientB));
    }

    // ------------------------------------------------------------------
    // Permission edge cases
    // ------------------------------------------------------------------

    function test_RevertWhen_NonCoordinatorOpensRound() public {
        vm.prank(stranger);
        vm.expectRevert(FLAggregator.NotCoordinator.selector);
        agg.openRound(BASE_REF, 1 days);
    }

    function test_RevertWhen_UnregisteredClientSubmits() public {
        agg.openRound(BASE_REF, 1 days);
        vm.prank(stranger);
        vm.expectRevert(FLAggregator.NotClient.selector);
        agg.submitUpdate(UPDATE_A);
    }

    function test_RevertWhen_NonCoordinatorFinalises() public {
        agg.openRound(BASE_REF, 1 days);
        vm.prank(clientA);
        vm.expectRevert(FLAggregator.NotCoordinator.selector);
        agg.finaliseRound(GLOBAL_REF);
    }

    function test_RevertWhen_DoubleSubmission() public {
        agg.openRound(BASE_REF, 1 days);
        vm.prank(clientA);
        agg.submitUpdate(UPDATE_A);
        vm.prank(clientA);
        vm.expectRevert(FLAggregator.AlreadySubmitted.selector);
        agg.submitUpdate(UPDATE_B);
    }

    function test_RevertWhen_LateSubmission() public {
        agg.openRound(BASE_REF, 1 hours);
        vm.warp(block.timestamp + 2 hours);
        vm.prank(clientA);
        vm.expectRevert(FLAggregator.DeadlinePassed.selector);
        agg.submitUpdate(UPDATE_A);
    }

    function test_RevertWhen_SubmitWithNoOpenRound() public {
        vm.prank(clientA);
        vm.expectRevert(FLAggregator.RoundNotOpen.selector);
        agg.submitUpdate(UPDATE_A);
    }

    function test_RevertWhen_OpenWhilePreviousRoundOpen() public {
        agg.openRound(BASE_REF, 1 days);
        vm.expectRevert(FLAggregator.RoundStillOpen.selector);
        agg.openRound(BASE_REF, 1 days);
    }

    function test_RevertWhen_DoubleFinalise() public {
        agg.openRound(BASE_REF, 1 days);
        agg.finaliseRound(GLOBAL_REF);
        vm.expectRevert(FLAggregator.RoundNotOpen.selector);
        agg.finaliseRound(GLOBAL_REF);
    }

    function test_RevertWhen_EmptyRefs() public {
        vm.expectRevert(FLAggregator.EmptyRef.selector);
        agg.openRound(bytes32(0), 1 days);
    }

    // ------------------------------------------------------------------
    // Gas bounds on hot paths (fail the build on regression)
    // ------------------------------------------------------------------

    function test_GasBound_SubmitUpdate() public {
        agg.openRound(BASE_REF, 1 days);
        vm.prank(clientA);
        uint256 g0 = gasleft();
        agg.submitUpdate(UPDATE_A);
        uint256 used = g0 - gasleft();
        emit log_named_uint("submitUpdate gas", used);
        assertLt(used, MAX_GAS_SUBMIT);
    }

    function test_GasBound_FinaliseRound() public {
        agg.openRound(BASE_REF, 1 days);
        vm.prank(clientA);
        agg.submitUpdate(UPDATE_A);
        uint256 g0 = gasleft();
        agg.finaliseRound(GLOBAL_REF);
        uint256 used = g0 - gasleft();
        emit log_named_uint("finaliseRound gas", used);
        assertLt(used, MAX_GAS_FINALISE);
    }

    // ------------------------------------------------------------------
    // Fuzz: any registered client can submit any non-zero ref once
    // ------------------------------------------------------------------

    function testFuzz_SubmitStoresRef(bytes32 ref) public {
        vm.assume(ref != bytes32(0));
        agg.openRound(BASE_REF, 1 days);
        vm.prank(clientA);
        agg.submitUpdate(ref);
        assertEq(agg.getUpdateRef(1, clientA), ref);
    }
}
