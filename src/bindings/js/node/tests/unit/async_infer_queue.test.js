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
  });

  function basicUserCallback(request, jobId, err) {
    if (err) {
      console.error(`Job ${jobId} failed: ${err}`);
    } else {
      assert.strictEqual(request instanceof ov.InferRequest, true);
    }
  }

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

  it('Test AsyncInferQueue constructor with invalid arguments', async () => {
    assert.throws(() => {
      new ov.AsyncInferQueue(); // No arguments
    },
    /'AsyncInferQueue' constructor method called with incorrect parameters./
    );

    assert.throws(() => {
      new ov.AsyncInferQueue(compiledModel); // Missing numRequest
    },
    /'AsyncInferQueue' constructor method called with incorrect parameters./
    );

    assert.throws(() => {
      new ov.AsyncInferQueue(null, numRequest); // Invalid compiledModel
    }, 'TypeError: Cannot convert undefined or null to object');

    assert.throws(() => {
      // Invalid numRequest type
      new ov.AsyncInferQueue(compiledModel, 'invalid');
    },
    /'AsyncInferQueue' constructor method called with incorrect parameters./
    );
  });

  it('Test main event loop non-blocking inference', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    const jobsDone = Array.from({ length: jobs }, () => ({ finished: false }));

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        jobsDone[jobId].finished = true;
        assert.ok(request instanceof ov.InferRequest);
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
    inferQueue.release();
  });

  it('test Promise.all() ~ infer_queue.wait_all()', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    const jobsDone = Array.from({ length: jobs }, () => ({ finished: false }));

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        jobsDone[jobId].finished = true;
        const inferenceResult = request.getOutputTensor().data;
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
    inferQueue.release();
  });

  it('Test AsyncInferQueue no freeze', async () => {
    try {
      new ov.AsyncInferQueue(compiledModel, numRequest);
    } catch(err) {
      assert.fail(`Unexpected error thrown: ${err.message}`);
    }
  });

  it('Test AsyncInferQueue.release()', async () => {
    try {
      const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
      inferQueue.release();
    } catch(err) {
      assert.fail(`Unexpected error thrown: ${err.message}`);
    }
  });

  it('Test repeated AsyncInferQueue.release()', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    inferQueue.setCallback(basicUserCallback);
    inferQueue.release();
    assert.throws(() => {
      inferQueue.release();
    }, /Failed to release AsyncInferQueue thread-safe function./);

  });

  it('Test setCallback throws and list possible signatures', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    assert.throws(() => {
      inferQueue.setCallback();
    }, /'set_callback' method called with incorrect parameters./);
    inferQueue.release();
  });

  it('Test startAsync throws and list possible signatures', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);
    inferQueue.setCallback(basicUserCallback);
    assert.throws(() => {
      inferQueue.startAsync({ 'data': generateImage() });
    }, /'startAsync' method called with incorrect parameters./);
    inferQueue.release();
  });

  it('Test possibility to catch error in callback', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        assert.ok(request instanceof ov.InferRequest);
        assert.throws(() => {
          // eslint-disable-next-line no-undef
          jobsDone[jobId].finished = true;
        }, /ReferenceError: jobsDone is not defined/);

      }
    }
    inferQueue.setCallback(callback);
    await inferQueue.startAsync({ 'data': generateImage() }, 'data');
    inferQueue.release();
  });

  it('Test error in callback and rejected promise', async () => {
    const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);

    function callback(request, jobId, err) {
      if (err) {
        console.error(`Job ${jobId} failed: ${err}`);
      } else {
        assert.ok(request instanceof ov.InferRequest);
        // eslint-disable-next-line no-undef
        jobsDone[jobId].finished = true; // throws ReferenceError
      }
    }

    inferQueue.setCallback(callback);
    await inferQueue.startAsync({ 'data': generateImage() }, 'data')
      .catch((err) => {
        assert(err instanceof Error);
        assert.strictEqual(err.message, 'jobsDone is not defined');
      });
    inferQueue.release();
  });

});
