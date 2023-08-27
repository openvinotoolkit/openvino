// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

describe('InferRequest', () => {

  let testXml = getModelPath();
  const core = new ov.Core();
  const model = core.readModel(testXml);
  const compiledModel = core.compileModel(model, 'CPU');

  const inferRequest = compiledModel.createInferRequest();

  const tensorData = Float32Array.from({ length: 3072 }, () => Math.random());
  const tensorData2 = Float32Array.from({ length: 3072 }, () => Math.random());
  assert.notDeepStrictEqual(tensorData, tensorData2);
  const tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 32, 32]),
    tensorData,
  );
  const tensor2 = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 32, 32]),
    tensorData2,
  );
  const res_tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 10]),
    Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
  );

  const tensorLike = [[tensor],
    [tensorData]];

  tensorLike.forEach(([tl]) => {
    inferRequest.infer({ data: tl });
    const result = inferRequest.getOutputTensors();
    const label = tl instanceof Float32Array ? 'TypedArray' : 'Tensor';
    it(`Test infer(inputData: { [inputName: string]: ${label} })`, () => {
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);
    });
  });

  tensorLike.forEach(([tl]) => {
    inferRequest.infer([tl]);
    const result = inferRequest.getOutputTensors();
    const label = tl instanceof Float32Array ? 'TypedArray' : 'Tensor';
    it(`Test infer(inputData: [ [inputName: string]: ${label} ])`, () => {
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);
    });
  });

  it('Test setInputTensor(tensor)', () => {
    inferRequest.setInputTensor(tensor);
    const t1 = inferRequest.getInputTensor();
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test setInputTensor(idx, tensor)', () => {
    inferRequest.setInputTensor(0, tensor);
    const t1 = inferRequest.getInputTensor();
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test setOutputTensor(tensor)', () => {
    inferRequest.setOutputTensor(res_tensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(res_tensor.data[0], res2.data[0]);
  });

  it('Test setOutputTensor(idx, tensor)', () => {
    inferRequest.setOutputTensor(0, res_tensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(res_tensor.data[0], res2.data[0]);
  });

  it('Test setTensor(string, tensor)', () => {
    inferRequest.setTensor('fc_out', res_tensor);
    const t1 = inferRequest.getTensor('fc_out');
    assert.deepStrictEqual(res_tensor.data[0], t1.data[0]);
  });

  it('Test getInputTensor(idx)', () => {
    inferRequest.setInputTensor(tensor2);
    const t2 = inferRequest.getInputTensor(0);
    assert(Math.abs(tensor2.data[0] - t2.data[0]) < 0.0001);
  });

  it('Test getTensor(string)', () => {
    inferRequest.setInputTensor(tensor);
    const t1 = inferRequest.getTensor('data');
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test getTensor(Output)', () => {
    inferRequest.setInputTensor(tensor2);
    const input = inferRequest.getCompiledModel().input();
    const t2 = inferRequest.getTensor(input);
    assert(Math.abs(tensor2.data[0] - t2.data[0]) < 0.0001);
  });

  it('Test getOutputTensor(idx?)', () => {
    const ir = compiledModel.createInferRequest();
    ir.setInputTensor(tensor2);
    ir.infer();
    const res1 = ir.getOutputTensor();
    const res2 = ir.getOutputTensor(0);
    assert(Math.abs(res2.data[0] - res1.data[0]) < 0.0001);
  });

  it('Test getCompiledModel()', () => {
    const ir = compiledModel.createInferRequest();
    const cm = ir.getCompiledModel();
    const ir2 = cm.createInferRequest();
    const res2 = ir2.infer([tensorData]);
    const res1 = ir.infer([tensorData]);
    // assert(instanceOf(compiledMode, cm));
    assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
  });
});
