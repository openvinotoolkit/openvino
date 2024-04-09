// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('..');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

const epsilon = 0.5; // To avoid very small numbers
const testXml = getModelPath().xml;
const core = new ov.Core();
const model = core.readModelSync(testXml);
const compiledModel = core.compileModelSync(model, 'CPU');

const inferRequest = compiledModel.createInferRequest();
const inferRequestAsync = compiledModel.createInferRequest();

const tensorData = Float32Array.from({ length: 3072 }, () => (Math.random() + epsilon));
const tensor = new ov.Tensor(
  ov.element.f32,
  [1, 3, 32, 32],
  tensorData,
);
const resTensor = new ov.Tensor(
  ov.element.f32,
  [1, 10],
  tensorData.slice(-10),
);

const tensorLike = [[tensor],
  [tensorData]];

describe('InferRequest', () => {

  tensorLike.forEach(([tl]) => {
    const result = inferRequest.infer({ data: tl });
    const label = tl instanceof Float32Array ? 'TypedArray[]' : 'Tensor[]';
    it(`Test infer(inputData: { [inputName: string]: ${label} })`, () => {
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);
    });
  });

  tensorLike.forEach(([tl]) => {
    const result = inferRequest.infer([tl]);
    const label = tl instanceof Float32Array ? 'TypedArray[]' : 'Tensor[]';
    it(`Test infer(inputData: ${label})`, () => {
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);
    });

  });

  it('Test infer(TypedArray) throws', () => {
    assert.throws(
      () => inferRequest.infer(tensorData),
      {message: 'TypedArray cannot be passed directly into infer() method.'});
  });

  const buffer = new ArrayBuffer(tensorData.length);
  const inputMessagePairs = [
    ['string', 'Cannot create a tensor from the passed Napi::Value.'],
    [tensorData.slice(-10), 'Memory allocated using shape and element::type mismatch passed data\'s size'],
    [new Float32Array(buffer, 4), 'TypedArray.byteOffset has to be equal to zero.'],
    [{}, /Invalid argument/], // Test for object that is not Tensor
  ];

  inputMessagePairs.forEach( ([tl, msg]) => {
    it(`Test infer([data]) throws ${msg}`, () => {
      assert.throws(
        () => inferRequest.infer([tl]),
        {message: new RegExp(msg)});
    });
    it(`Test infer({ data: tl}) throws ${msg}`, () => {
      assert.throws(
        () => inferRequest.infer({data: tl}),
        {message: new RegExp(msg)});
    });
  });

  it('Test inferAsync(inputData: { [inputName: string]: Tensor })', () => {
    inferRequestAsync.inferAsync({ data: tensor }).then(result => {
      assert.ok(result['fc_out'] instanceof ov.Tensor);
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);}
    );
  });

  it('Test inferAsync(inputData: Tensor[])', () => {
    inferRequestAsync.inferAsync([ tensor ]).then(result => {
      assert.ok(result['fc_out'] instanceof ov.Tensor);
      assert.deepStrictEqual(Object.keys(result), ['fc_out']);
      assert.deepStrictEqual(result['fc_out'].data.length, 10);
    });
  });

  it('Test inferAsync([data]) throws: Cannot create a tensor from the passed Napi::Value.', () => {
    assert.throws(
      () => inferRequestAsync.inferAsync(['string']).then(),
      /Cannot create a tensor from the passed Napi::Value./
    );
  });

  it('Test inferAsync({ data: "string"}) throws: Cannot create a tensor from the passed Napi::Value.', () => {
    assert.throws(
      () => inferRequestAsync.inferAsync({data: 'string'}).then(),
      /Cannot create a tensor from the passed Napi::Value./
    );
  });

  it('Test setInputTensor(tensor)', () => {
    inferRequest.setInputTensor(tensor);
    const t1 = inferRequest.getInputTensor();
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test setInputTensor(object) throws when passed object is not a Tensor.', () => {
    assert.throws(
      () => inferRequest.setInputTensor({}),
      {message: /Argument #[0-9]+ must be a Tensor./}
    );
  });

  it('Test setInputTensor(idx, tensor)', () => {
    inferRequest.setInputTensor(0, tensor);
    const t1 = inferRequest.getInputTensor();
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test setInputTensor(idx, tensor) throws', () => {
    const testIdx = 10;
    assert.throws (
      () => inferRequest.setInputTensor(testIdx, tensor),
      {message: /Input port for index [0-9]+ was not found!/}
    );
  });

  it('Test setInputTensor(idx, object) throws when passed object is not a Tensor.', () => {
    assert.throws(
      () => inferRequest.setInputTensor(0, {}),
      {message: /Argument #[0-9]+ must be a Tensor./}
    );
  });

  it('Test setInputTensor(tensor, tensor) throws', () => {
    assert.throws(
      () => inferRequest.setInputTensor(resTensor, tensor),
      {message: / invalid argument./}
    );
  });

  it('Test setOutputTensor(tensor)', () => {
    inferRequest.setOutputTensor(resTensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test setOutputTensor(object) throws when passed object is not a Tensor.', () => {
    assert.throws(
      () => inferRequest.setOutputTensor({}),
      {message: /Argument #[0-9]+ must be a Tensor./}
    );
  });

  it('Test setOutputTensor(idx, tensor) throws', () => {
    const testIdx = 10;
    assert.throws (
      () => inferRequest.setOutputTensor(testIdx, tensor),
      {message: /Output port for index [0-9]+ was not found!/}
    );
  });

  it('Test setOutputTensor(idx, tensor)', () => {
    inferRequest.setOutputTensor(0, resTensor);
    const res2 = inferRequest.getOutputTensor();
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test setOutputTensor(idx, tensor) throws when passed object is not a Tensor.', () => {
    assert.throws(
      () => inferRequest.setOutputTensor(0, {}),
      {message: /Argument #[0-9]+ must be a Tensor./}
    );
  });

  it('Test setOutputTensor() - pass two tensors', () => {
    assert.throws(
      () => inferRequest.setOutputTensor(resTensor, tensor),
      {message: / invalid argument./});
  });

  it('Test setTensor(string, tensor)', () => {
    inferRequest.setTensor('fc_out', resTensor);
    const res2 = inferRequest.getTensor('fc_out');
    assert.ok(res2 instanceof ov.Tensor);
    assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
  });

  it('Test setTensor(string, object) - throws', () => {
    const testName = 'testName';
    assert.throws(
      () => inferRequest.setTensor(testName, tensor),
      {message: /Port for tensor name testName was not found./});
  });

  it('Test setTensor(string, object) - throws', () => {
    assert.throws(
      () => inferRequest.setTensor('fc_out', {}),
      {message: /Argument #[0-9]+ must be a Tensor./});
  });

  it('Test setTensor(string, tensor) - pass one arg', () => {
    assert.throws(
      () => inferRequest.setTensor('fc_out'),
      {message: / invalid argument./}
    );
  });

  it('Test setTensor(string, tensor) - pass args in wrong order', () => {
    assert.throws(
      () => inferRequest.setTensor(resTensor, 'fc_out'),
      {message: / invalid argument./}
    );
  });

  it('Test setTensor(string, tensor) - pass number as first arg', () => {
    assert.throws(
      () => inferRequest.setTensor(123, 'fc_out'),
      {message: / invalid argument/}
    );
  });

  const irGetters = compiledModel.createInferRequest();
  irGetters.setInputTensor(tensor);
  irGetters.infer();

  it('Test getTensor(tensorName)', () => {
    const t1 = irGetters.getTensor('data');
    assert.ok(t1 instanceof ov.Tensor);
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test getTensor(Output)', () => {
    const input = irGetters.getCompiledModel().input();
    const t1 = irGetters.getTensor(input);
    assert.ok(t1 instanceof ov.Tensor);
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test getInputTensor()', () => {
    const t1 = irGetters.getInputTensor();
    assert.ok(t1 instanceof ov.Tensor);
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test getInputTensor(idx)', () => {
    const t1 = irGetters.getInputTensor(0);
    assert.ok(t1 instanceof ov.Tensor);
    assert.deepStrictEqual(tensor.data[0], t1.data[0]);
  });

  it('Test getOutputTensor(idx?)', () => {
    const res1 = irGetters.getOutputTensor();
    const res2 = irGetters.getOutputTensor(0);
    assert.ok(res1 instanceof ov.Tensor);
    assert.ok(res2 instanceof ov.Tensor);
    assert.deepStrictEqual(res1.data[0], res2.data[0]);
  });

  it('Test getCompiledModel()', () => {
    const ir = compiledModel.createInferRequest();
    const cm = ir.getCompiledModel();
    assert.ok(cm instanceof ov.CompiledModel);
    const ir2 = cm.createInferRequest();
    const res2 = ir2.infer([tensorData]);
    const res1 = ir.infer([tensorData]);
    assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
  });
});
