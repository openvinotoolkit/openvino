// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

describe('InferRequest', () => {

  const testXml = getModelPath();
  const core = new ov.Core();
  const model = core.readModel(testXml);
  const compiledModel = core.compileModel(model, 'CPU');

  const inferRequest = compiledModel.createInferRequest();

  const tensorData = Float32Array.from({ length: 3072 }, () => Math.floor(Math.random() * 3072));
  const tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 32, 32]),
    tensorData,
  );
  const resTensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 10]),
    tensorData.slice(-10),
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
    inferRequest.setOutputTensor(resTensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test setOutputTensor(idx, tensor)', () => {
    inferRequest.setOutputTensor(0, resTensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test setTensor(string, tensor)', () => {
    inferRequest.setTensor('fc_out', resTensor);
    const res2 = inferRequest.getTensor('fc_out');
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test of getters', () => {
    const ir = compiledModel.createInferRequest();
    ir.setInputTensor(tensor);

    const t1 = ir.getInputTensor(0);
    const t2 = ir.getTensor('data');
    const input = ir.getCompiledModel().input();
    const t3 = ir.getTensor(input);

    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    assert.deepStrictEqual(tensor.data[0], t2.data[0]);
    assert.deepStrictEqual(tensor.data[0], t3.data[0]);

    ir.infer();
    const res1 = ir.getOutputTensor();
    const res2 = ir.getOutputTensor(0);
    assert.deepStrictEqual(res1.data[0], res2.data[0]);
  });

  it('Test getCompiledModel()', () => {
    const ir = compiledModel.createInferRequest();
    const cm = ir.getCompiledModel();
    const ir2 = cm.createInferRequest();
    const res2 = ir2.infer([tensorData]);
    const res1 = ir.infer([tensorData]);
    // assert(instanceOf(compiledMode, cm)); // TODO Create a separate test
    assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
  });
});
