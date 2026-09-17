// A ready-made upload file: two batches of ordinary payments, with one wash-
// trading loop hidden in the first and one flash-loan burst in the second.
//
// Ordinary payments come from a seeded Park-Miller generator rather than a
// fixed pattern: a regular pattern forms artificial loops between ordinary
// wallets, which the model rightly finds suspicious, and the example would then
// show false alarms that no real data would produce. The multiplier is small
// enough that every step is exact in a JavaScript double.

const WALLETS = 40;
const PAYMENTS_PER_BATCH = 60;

function generator(seed) {
  let state = seed;
  return () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  };
}

const pad = (n) => String(n).padStart(2, "0");

export function exampleCsv(seed = 42) {
  const rand = generator(seed);
  const rows = ["from,to,value,block,is_fraud,fraud_type"];

  for (const block of [1, 2]) {
    let made = 0;
    while (made < PAYMENTS_PER_BATCH) {
      const a = Math.floor(rand() * WALLETS) + 1;
      const b = Math.floor(rand() * WALLETS) + 1;
      const value = 0.6 + rand() * 0.8;
      if (a === b) continue;
      rows.push(`wallet-${pad(a)},wallet-${pad(b)},${value.toFixed(2)},${block},0,`);
      made += 1;
    }
  }

  const loop = ["wash-A", "wash-B", "wash-C", "wash-D"];
  loop.forEach((w, i) => rows.push(`${w},${loop[(i + 1) % loop.length]},5,1,1,wash`));

  for (let i = 1; i <= 25; i += 1) {
    rows.push(`flash-X,pool-${pad(i)},50,2,1,flash`);
    rows.push(`pool-${pad(i)},flash-X,50,2,0,`);
  }

  return rows.join("\n");
}
