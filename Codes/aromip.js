#!/usr/bin/env node

// AroMIP: Aromatic Membrane Insertion Predictor
// Usage: node aromip.js <sequence>
// Example: node aromip.js RRNKFGINRTTGNWRGMLQRDLYSGLN

function Sigmoid(Q) {
  if (Q <= 0) return 0.0;
  return Q / (1 + Q);
}

function modelPrediction(fragment, paramData) {
  const { params } = paramData;
  let product = 1.0;
  for (let i = 0; i < fragment.length; i++) {
    if (i === 4) continue;
    const aa = fragment[i];
    if (!(aa in params)) continue;
    product *= Math.max(1e-8, params[aa]);
  }
  return { Q: product, score: Sigmoid(product) };
}

function scanProteinSequence(sequence) {
  const results = [];
  const centerAASet = new Set(["F", "W", "Y"]);
  sequence = sequence.toUpperCase();

  const paramSets = {
    F: {
      params: {
        A: 1.0758068943930996, C: 0.7515081115208946, D: 0.4098861523565832,
        E: 0.3983119844300048, F: 8.1603925477323100, G: 0.6354157796571205,
        H: 0.7846691451344266, I: 3.1491216060357460, K: 1.1009251977458419,
        L: 3.6046544612488423, M: 2.8653319073560772, N: 0.4803481572211203,
        P: 1.7694984669210487, Q: 0.4933503638424146, R: 1.6585225219886708,
        S: 0.9301680536402084, T: 1.2417197033480494, V: 2.0355593428760814,
        W: 5.9063068700289690, Y: 1.8801996532692813
      }
    },
    W: {
      params: {
        A: 1.0206442786302903, C: 0.6708211019772945, D: 0.3925586374549908,
        E: 0.4000948935406380, F: 12.4315596752144760, G: 0.5956514706239270,
        H: 0.7191760324420514, I: 2.9251386801066905, K: 1.0112142249802383,
        L: 4.0861358235697650, M: 2.8506481939604393, N: 0.4732539101826204,
        P: 1.6093272013276745, Q: 0.4767325148760477, R: 1.4493178760361400,
        S: 0.9053164338059225, T: 1.1254876583549318, V: 2.0417309116628624,
        W: 11.5571567222502710, Y: 1.9236800785696822
      }
    },
    Y: {
      params: {
        A: 0.6230184781505200, C: 0.0133352143216332, D: 0.2214359322487689,
        E: 0.3402031615640565, F: 2.1219401227664230, G: 0.3466014548722871,
        H: 0.2072649540790862, I: 1.7025466417820883, K: 0.5832851280298242,
        L: 1.5741986228306597, M: 0.9020719922136957, N: 0.2975325873058723,
        P: 0.9559854615896372, Q: 0.4220647091504225, R: 1.1369649667516084,
        S: 0.4189210011424207, T: 0.5604827271090070, V: 1.0606855747783748,
        W: 1.7911226699514830, Y: 1.2730210580996608
      }
    }
  };

  for (let i = 4; i < sequence.length - 4; i++) {
    const fragment = sequence.slice(i - 4, i + 5);
    const centerAA = fragment[4];
    if (!centerAASet.has(centerAA)) continue;
    const { Q, score } = modelPrediction(fragment, paramSets[centerAA]);
    results.push({
      fragment,
      score: score.toFixed(3),
      classification: score > 0.5 ? "Inserter" : "Non-inserter"
    });
  }

  return results;
}

// --- CLI entry point ---
const args = process.argv.slice(2);

if (args.length === 0) {
  console.error("Usage: node aromip.js <sequence>");
  console.error("Example: node aromip.js RRNKFGINRTTGNWRGMLQRDLYSGLN");
  process.exit(1);
}

const sequence = args[0];
const results = scanProteinSequence(sequence);

if (results.length === 0) {
  console.log("No valid 9-residue fragments found centered on F, W, or Y.");
  process.exit(0);
}

// Print header
const col1 = "Motif".padEnd(12);
const col2 = "P (insertion)".padEnd(16);
const col3 = "Prediction";
console.log(`${col1}${col2}${col3}`);
console.log("-".repeat(col1.length + col2.length + col3.length));

for (const r of results) {
  console.log(`${r.fragment.padEnd(12)}${r.score.padEnd(16)}${r.classification}`);
}
