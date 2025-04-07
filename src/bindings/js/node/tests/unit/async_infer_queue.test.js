// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('../..');
const { describe, it} = require('node:test');

describe('Tests for AsyncInferQueue.', () => {

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

  describe('AsyncInferQueue python api scenario', () => {

    it('https://github.com/openvinotoolkit/openvino/blob/6d23ba5aae91fc7c8848a4d03db50a7655490135/src/bindings/python/tests/test_runtime/test_async_infer_request.py#L31', async () => {
      const jobs = 8;
      const numRequest = 4;
      const core = new ov.Core();
      const model = await core.readModel('model.xml');
      const compiledModel = core.compileModelSync(model, 'CPU');
      const inferQueue = new ov.AsyncInferQueue(compiledModel, numRequest);

      function callback(request, jobId, err) {
        if (err) {
          console.error(`Job ${jobId} failed: ${err}`);
        } else {
          console.log(`Job ${jobId} finished successfully`);
          const result = request.getOutputTensor().data;
          console.log(Array.from(result).slice(0, 3), jobId);
        }
      }

      inferQueue.setCallback(callback);

      for (let i = 0; i < jobs; i++) {
        // await new Promise(resolve => setTimeout(resolve, 1000));
        const img = generateImage();
        inferQueue.startAsync({'data': img}, i);
      }

      inferQueue.waitAll(); // TODO fix
      console.log('end of test');

    });

  });

});
