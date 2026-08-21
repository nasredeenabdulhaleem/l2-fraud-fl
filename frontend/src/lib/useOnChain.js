import { useEffect, useRef, useState } from "react";
import { Contract, JsonRpcProvider } from "ethers";

const RPC_URL = import.meta.env.VITE_RPC_URL || "";
const CONTRACT_ADDRESS = import.meta.env.VITE_FLAGGREGATOR_ADDRESS || "";
const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";
const POLL_MS = 5000;

const ABI = [
  "function currentRoundId() view returns (uint64)",
  "function getRound(uint64 roundId) view returns (tuple(bytes32 baseModelRef, bytes32 globalModelRef, uint64 deadline, uint32 submissionCount, bool open, bool finalised))",
];

/**
 * Reads FLAggregator directly from the browser via a read-only RPC call, as an
 * independent check on whatever the WebSocket telemetry feed reported -- if
 * both agree, the dashboard isn't just trusting the backend's word for it.
 */
export function useOnChain() {
  const [state, setState] = useState({
    roundId: null,
    globalModelRef: null,
    finalised: false,
    error: null,
  });
  const contractRef = useRef(null);

  useEffect(() => {
    if (!RPC_URL || !CONTRACT_ADDRESS || CONTRACT_ADDRESS === ZERO_ADDRESS) {
      setState((s) => ({ ...s, error: "not configured" }));
      return;
    }

    const provider = new JsonRpcProvider(RPC_URL);
    contractRef.current = new Contract(CONTRACT_ADDRESS, ABI, provider);

    let stopped = false;

    const poll = async () => {
      try {
        const roundId = await contractRef.current.currentRoundId();
        if (roundId === 0n) {
          if (!stopped) setState({ roundId: 0, globalModelRef: null, finalised: false, error: null });
          return;
        }
        const round = await contractRef.current.getRound(roundId);
        if (!stopped) {
          setState({
            roundId: Number(roundId),
            globalModelRef: round.globalModelRef,
            finalised: round.finalised,
            error: null,
          });
        }
      } catch (err) {
        if (!stopped) setState((s) => ({ ...s, error: err.message || "read failed" }));
      }
    };

    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, []);

  return state;
}
