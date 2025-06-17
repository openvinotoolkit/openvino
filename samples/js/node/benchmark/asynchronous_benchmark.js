// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* global BigInt */

const { addon: ov } = require('openvino-node');
const path = require('path');
const { hrtime } = require('process');

function generateInputData(tensor) {
  const shape = tensor.getShape();
  const type = tensor.getElementType();

  const numElements = shape.reduce(
    (accumulator, currentValue) => accumulator * currentValue,
    1,
  );

  const epsilon = 0.5; // To avoid very small numbers
  const tensorData = Float32Array.from(
    { length: numElements },
    () => Math.random() + epsilon,
  );

  return new ov.Tensor(type, shape, tensorData);
}

async function main() {
  let deviceName = 'CPU';
  const args = process.argv;

  if (args.length === 4) {
    deviceName = args[2];
  } else if (args.length !== 3) {
    console.log(`Usage: ${path.basename(args[1])} 
      <path_to_model> <device_name>(default: CPU)`);

    process.exit(1);
  }
  // Optimize for throughput.
  const tput = {'PERFORMANCE_HINT': 'THROUGHPUT'};

  const core = new ov.Core();
  // Reads and compiles the model with one input and one output.
  const compiledModel = await core.compileModel(args[2], deviceName, tput);

  const { latencies, elapsed } = await benchmark(compiledModel);

  const count = latencies.length;
  const elapsedMs = Number(elapsed) / 1e6;
  const fps = count / (elapsedMs / 1000);

  const avgTime = latencies.reduce((a, b) => a + b, 0) / count;
  const minTime = Math.min(...latencies);
  const maxTime = Math.max(...latencies);
  const sortedTimes = [...latencies].sort((a, b) => a - b);
  const mid = Math.floor(sortedTimes.length / 2);
  const medianTime = sortedTimes.length % 2 !== 0
    ? sortedTimes[mid]
    : (sortedTimes[mid - 1] + sortedTimes[mid]) / 2;

  console.log(`Count:          ${count} iterations`);
  console.log(`Duration:       ${elapsedMs.toFixed(2)} ms`);
  console.log('Latency:');
  console.log(`    Median:     ${medianTime.toFixed(2)} ms`);
  console.log(`    Average:    ${avgTime.toFixed(2)} ms`);
  console.log(`    Min:        ${minTime.toFixed(2)} ms`);
  console.log(`    Max:        ${maxTime.toFixed(2)} ms`);
  console.log(`Throughput: ${fps.toFixed(2)} FPS`);

}

async function benchmark(model) {
  const inferRequest = model.createInferRequest();
  const tensor = generateInputData(inferRequest.getInputTensor());

  // Warm up
  for (let i = 0; i < 10; i++) {
    await inferRequest.inferAsync([tensor]);
  }
  // Benchmark for seconds_to_run seconds and at least niter iterations
  const minSeconds = 10;
  const minIter = 10;
  const latencies = [];
  const start = hrtime.bigint();
  let elapsed = 0n;
  let iterations = 0;
  while (elapsed < BigInt(minSeconds) * BigInt(1e9) || iterations < minIter) {
    const iterStart = hrtime.bigint();
    // Performs inference and does not block the event loop.
    await inferRequest.inferAsync([tensor]);
    latencies.push(Number(hrtime.bigint() - iterStart) / 1e6);
    elapsed = hrtime.bigint() - start;
    iterations++;
  }
  elapsed = hrtime.bigint() - start;

  return { latencies, elapsed };
}

main();
