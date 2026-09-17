// FastAPI reports errors in two different shapes: a plain string for the ones
// the handlers raise themselves (HTTPException(detail="...")), and an array of
// per-field objects for request-validation failures. Interpolating the second
// into a template gives "[object Object]", so normalise both to a readable
// string here rather than at each call site.
export function apiErrorMessage(payload, fallback) {
  const detail = payload?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const parts = detail.map((entry) => {
      // loc is a path like ["body", "threshold"]; "body" is noise to a reader.
      const field = (entry.loc || []).filter((p) => p !== "body").join(".");
      return field ? `${field}: ${entry.msg}` : entry.msg;
    });
    if (parts.length) return parts.join("; ");
  }

  return fallback;
}
