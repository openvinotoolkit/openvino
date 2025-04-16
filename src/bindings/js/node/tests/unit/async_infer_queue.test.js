// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const assert = require('assert');
const { addon: ov } = require('../..');
const { describe, it, before } = require('node:test');
const {
  getReluModel,
} = require('../utils.js');

describe('Tests for AsyncInferQueue.', () => {
  const jobs = 8;
  const numRequest = 4;
  let core = null;
  let compiledModel = null;

  before(async () => {
    core = new ov.Core();
    const model = await core.readModel(getReluModel());
    compiledModel = core.compileModelSync(model, 'CPU');
  }
  );

  // TODO move it to utils and use it in other tests
  function generateImage(shape = [1, 3, 32, 32]) {
    const lemm = shape.reduce(
      (accumulator, currentValue) => accumulator * currentValue,
      1
    );

    const epsilon = 0.5; // To avoid very small numbers
    const tensorData = Float32Array.from(
      { length: lemm },
      () => Math.random() + epsilon,
    );

    return tensorData;
  }

  it('Test main event loop non-blocking inference', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    const jobsDone = Array.from({ length: jobs }, () => ({ finished: false }));

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        jobsDone[jobId].finished = true;
        const result = request.getOutputTensor().data;
        console.log(Array.from(result).slice(0, 3), jobId);
      }
    }

    inferQueue.setCallback(callback);

    for (let i = 0; i < jobs; i++) {
      const img = generateImage();
      // Start the inference request in non-blocking mode.
      // The results will be available in the callback function.
      await inferQueue.startAsync({ 'data': img }, i);
    }

    assert.strictEqual(jobsDone.filter(job => job.finished).length, jobs);
  });

  it('test Promise.all() ~ infer_queue.wait_all()', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    const jobsDone = Array.from({ length: jobs }, () => ({ finished: false }));

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        console.log(`Job ${jobId} finished`);
        jobsDone[jobId].finished = true;
        const inferenceResult = request.getOutputTensor().data;
        // TODO add test for catching errors from callback
        // e.g. using here i instead of jobId
        const inputAt0 = jobId * (jobId % 2 === 0 ? 1 : -1);
        const resultAt0 = inputAt0 > 0 ? inputAt0 : 0; // relu function
        assert.strictEqual(inferenceResult[0], resultAt0);
        const input = request.getInputTensor().data;
        assert.strictEqual(input[0], inputAt0);
      }
    }

    inferQueue.setCallback(callback);

    const promises = [];
    for (let i = 0; i < jobs; i++) {
      const img = generateImage();
      img[0] = i * (i % 2 === 0 ? 1 : -1);
      const promise = inferQueue.startAsync({ 'data': img }, i);
      promises.push(promise);
    }

    await Promise.all(promises);
    assert.strictEqual(jobsDone.filter(job => job.finished).length, jobs);
  });

});
