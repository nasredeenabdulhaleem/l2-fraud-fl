import { useEffect, useReducer, useRef, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

const initialState = {
  strategy: "fedprox",
  current_round: 0,
  total_rounds: 10,
  clients: [],
  round_history: [],
  fraud_alerts: [],
};

function reducer(state, msg) {
  switch (msg.type) {
    case "snapshot":
      return { ...state, ...msg.data };

    case "round_opened":
      return {
        ...state,
        current_round: msg.data.round,
        clients: state.clients.map((c) => ({ ...c, submitted: false })),
      };

    case "update_submitted":
      return {
        ...state,
        clients: state.clients.map((c) =>
          c.id === msg.data.client_id
            ? { ...c, submitted: true, fraud_rate: msg.data.fraud_rate }
            : c
        ),
      };

    case "fraud_flagged":
      return {
        ...state,
        fraud_alerts: [...state.fraud_alerts, msg.data].slice(-30),
      };

    case "round_finalised":
      return {
        ...state,
        round_history: [...state.round_history, msg.data].slice(-50),
      };

    default:
      return state;
  }
}

export function useTelemetry() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let retry;
    let stopped = false;

    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onmessage = (ev) => {
        try {
          dispatch(JSON.parse(ev.data));
        } catch (_) {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!stopped) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current && wsRef.current.close();
    };
  }, []);

  return { state, connected };
}
