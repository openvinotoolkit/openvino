// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const fs = require("node:fs/promises");
const { addon: ov } = require("../..");
const assert = require("assert");
const { describe, it, before, beforeEach } = require("node:test");
const { testModels, isModelAvailable, lengthFromShape, generateImage } = require("../utils.js");
const path = require("path");

describe("ov.InferRequest tests", () => {
  const { testModelFP32 } = testModels;
  let compiledModel = null;
  let tensorData = null;
  let tensor = null;
  let resTensor = null;
  let tensorLike = null;

  before(async () => {
    await isModelAvailable(testModelFP32);

    const core = new ov.Core();
    const model = core.readModelSync(testModelFP32.xml);
    compiledModel = core.compileModelSync(model, "CPU");

    tensorData = generateImage(testModelFP32.inputShape);
    tensor = new ov.Tensor(ov.element.f32, testModelFP32.inputShape, tensorData);
    resTensor = new ov.Tensor(ov.element.f32, testModelFP32.outputShape, tensorData.slice(-10));
    tensorLike = [tensor, tensorData];
  });

  describe("infer() method", () => {
    let inferRequest = null;
    beforeEach(() => {
      inferRequest = compiledModel.createInferRequest();
    });

    it("infer(inputData: {inputName: string]: Tensor[]/TypedArray[]}", () => {
      tensorLike.forEach((tl) => {
        const result = inferRequest.infer({ data: tl });
        assert.deepStrictEqual(Object.keys(result), ["fc_out"]);
        assert.deepStrictEqual(
          result["fc_out"].data.length,
          lengthFromShape(testModelFP32.outputShape),
        );
      });
    });

    it("Test infer(inputData: Tensor[]/TypedArray[])", () => {
      tensorLike.forEach((tl) => {
        const result = inferRequest.infer([tl]);
        assert.deepStrictEqual(Object.keys(result), ["fc_out"]);
        assert.deepStrictEqual(
          result["fc_out"].data.length,
          lengthFromShape(testModelFP32.outputShape),
        );
      });
    });

    it("Test infer(TypedArray) throws", () => {
      assert.throws(() => inferRequest.infer(tensorData), {
        message: /TypedArray cannot be passed directly into infer\(\) method./,
      });
    });

    it("Test for invalid input data", () => {
      const buffer = new ArrayBuffer(tensorData.length);
      const inputMessagePairs = [
        ["string", "Cannot create a tensor from the passed Napi::Value."],
        [tensorData.slice(-10), /Memory allocated using shape and element::type mismatch/],
        [new Float32Array(buffer, 4), "TypedArray.byteOffset has to be equal to zero."],
        [{}, /Invalid argument/], // Test for object that is not Tensor
      ];

      inputMessagePairs.forEach(([tl, msg]) => {
        assert.throws(
          () => inferRequest.infer([tl]),
          { message: new RegExp(msg) },
          "infer([data]) throws",
        );

        assert.throws(
          () => inferRequest.infer({ data: tl }),
          { message: new RegExp(msg) },
          "infer({ data: tl}) throws",
        );
      });
    });
  });

  describe("inferAsync() method", () => {
    let inferRequest = null;
    before(() => {
      inferRequest = compiledModel.createInferRequest();
    });

    it("Test inferAsync(inputData: { [inputName: string]: Tensor })", () => {
      inferRequest.inferAsync({ data: tensor }).then((result) => {
        assert.ok(result["fc_out"] instanceof ov.Tensor);
        assert.deepStrictEqual(Object.keys(result), ["fc_out"]);
        assert.deepStrictEqual(
          result["fc_out"].data.length,
          lengthFromShape(testModelFP32.outputShape),
        );
      });
    });

    it("Test inferAsync(inputData: Tensor[])", () => {
      inferRequest.inferAsync([tensor]).then((result) => {
        assert.ok(result["fc_out"] instanceof ov.Tensor);
        assert.deepStrictEqual(Object.keys(result), ["fc_out"]);
        assert.deepStrictEqual(
          result["fc_out"].data.length,
          lengthFromShape(testModelFP32.outputShape),
        );
      });
    });

    it("Test inferAsync([data]) throws", () => {
      assert.throws(
        () => inferRequest.inferAsync(["string"]).then(),
        /Cannot create a tensor from the passed Napi::Value./,
      );
    });

    it('Test inferAsync({ data: "string"}) throws', () => {
      assert.throws(
        () => inferRequest.inferAsync({ data: "string" }).then(),
        /Cannot create a tensor from the passed Napi::Value./,
      );
    });
  });

  describe("BigInt InferRequest support", () => {
    let core, originalModel, compiledModel, inferRequest;

    before(async () => {
      const { addModel } = testModels;
      await isModelAvailable(addModel);
      core = new ov.Core();
      originalModel = core.readModelSync(addModel.xml);
    });

    beforeEach(() => {
      const model = originalModel.clone();

      const ppp = new ov.preprocess.PrePostProcessor(model);
      ppp.input(0).tensor().setElementType(ov.element.i64);
      ppp.input(1).tensor().setElementType(ov.element.i64);
      ppp.build();

      compiledModel = core.compileModelSync(model, "CPU");
      inferRequest = compiledModel.createInferRequest();
    });

    it("infers with BigInt64Array input using Tensor objects", () => {
      const shape0 = originalModel.input(0).getShape();
      const shape1 = originalModel.input(1).getShape();
      const size0 = shape0.reduce((a, b) => a * b, 1);
      const size1 = shape1.reduce((a, b) => a * b, 1);
      const inputData0 = new BigInt64Array(size0).fill(1n);
      const inputData1 = new BigInt64Array(size1).fill(2n);

      const tensor0 = new ov.Tensor(ov.element.i64, shape0, inputData0);
      const tensor1 = new ov.Tensor(ov.element.i64, shape1, inputData1);

      assert.doesNotThrow(() => {
        const result = inferRequest.infer([tensor0, tensor1]);
        assert.ok(result);
        const outputTensor = Object.values(result)[0];
        assert.ok(outputTensor instanceof ov.Tensor);
      });
    });

    it("infers with BigInt64Array input using setInputTensor", () => {
      const shape0 = originalModel.input(0).getShape();
      const shape1 = originalModel.input(1).getShape();
      const size0 = shape0.reduce((a, b) => a * b, 1);
      const size1 = shape1.reduce((a, b) => a * b, 1);
      const inputData0 = new BigInt64Array(size0).fill(1n);
      const inputData1 = new BigInt64Array(size1).fill(2n);

      const tensor0 = new ov.Tensor(ov.element.i64, shape0, inputData0);
      const tensor1 = new ov.Tensor(ov.element.i64, shape1, inputData1);

      assert.doesNotThrow(() => {
        inferRequest.setInputTensor(0, tensor0);
        inferRequest.setInputTensor(1, tensor1);
        inferRequest.infer();
        const result = inferRequest.getOutputTensor();
        assert.ok(result instanceof ov.Tensor);
      });
    });

    it("errors on wrong BigInt64Array size", () => {
      const shape0 = originalModel.input(0).getShape();
      const size0 = shape0.reduce((a, b) => a * b, 1);
      const wrongSizeData = new BigInt64Array(Math.max(1, size0 - 1)).fill(0n);

      assert.throws(
        () => new ov.Tensor(ov.element.i64, shape0, wrongSizeData),
        /Memory allocated using shape and element::type mismatch/,
      );
    });

    it("infers with BigUint64Array input", () => {
      const model = originalModel.clone();
      const ppp = new ov.preprocess.PrePostProcessor(model);
      ppp.input(0).tensor().setElementType(ov.element.u64);
      ppp.input(1).tensor().setElementType(ov.element.u64);
      ppp.build();
      const compiledModelU64 = core.compileModelSync(model, "CPU");
      const inferRequestU64 = compiledModelU64.createInferRequest();

      const shape0 = originalModel.input(0).getShape();
      const shape1 = originalModel.input(1).getShape();
      const size0 = shape0.reduce((a, b) => a * b, 1);
      const size1 = shape1.reduce((a, b) => a * b, 1);
      const inputData0 = new BigUint64Array(size0).fill(1n);
      const inputData1 = new BigUint64Array(size1).fill(2n);

      const tensor0 = new ov.Tensor(ov.element.u64, shape0, inputData0);
      const tensor1 = new ov.Tensor(ov.element.u64, shape1, inputData1);

      assert.doesNotThrow(() => {
        const result = inferRequestU64.infer([tensor0, tensor1]);
        assert.ok(result);
        const outputTensor = Object.values(result)[0];
        assert.ok(outputTensor instanceof ov.Tensor);
      });
    });

    it("errors on wrong BigUint64Array size", () => {
      const shape0 = originalModel.input(0).getShape();
      const size0 = shape0.reduce((a, b) => a * b, 1);
      const wrongSizeData = new BigUint64Array(size0 + 1).fill(0n);

      assert.throws(
        () => new ov.Tensor(ov.element.u64, shape0, wrongSizeData),
        /Memory allocated using shape and element::type mismatch/,
      );
    });
  });
  describe("setters", () => {
    let inferRequest = null;
    beforeEach(() => {
      inferRequest = compiledModel.createInferRequest();
    });

    it("setInputTensor(tensor)", () => {
      inferRequest.setInputTensor(tensor);
      const t1 = inferRequest.getInputTensor();
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("setInputTensor(object) throws if object is not a Tensor.", () => {
      assert.throws(() => inferRequest.setInputTensor({}), {
        message: /Argument #[0-9]+ must be a Tensor./,
      });
    });

    it("setInputTensor(idx, tensor)", () => {
      inferRequest.setInputTensor(0, tensor);
      const t1 = inferRequest.getInputTensor();
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("setInputTensor(idx, tensor) throws", () => {
      const testIdx = 10;
      assert.throws(() => inferRequest.setInputTensor(testIdx, tensor), {
        message: /Input port for index [0-9]+ was not found!/,
      });
    });

    it("setInputTensor(idx, object) throws if object is not a Tensor.", () => {
      assert.throws(() => inferRequest.setInputTensor(0, {}), {
        message: /Argument #[0-9]+ must be a Tensor./,
      });
    });

    it("setInputTensor(tensor, tensor) throws", () => {
      assert.throws(() => inferRequest.setInputTensor(resTensor, tensor), {
        message: / invalid argument./,
      });
    });

    it("setOutputTensor(tensor)", () => {
      inferRequest.setOutputTensor(resTensor);
      const res2 = inferRequest.getOutputTensor();
      assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
    });

    it("setOutputTensor(object) throws if object is not a Tensor.", () => {
      assert.throws(() => inferRequest.setOutputTensor({}), {
        message: /Argument #[0-9]+ must be a Tensor./,
      });
    });

    it("setOutputTensor(idx, tensor) throws", () => {
      const testIdx = 10;
      assert.throws(() => inferRequest.setOutputTensor(testIdx, tensor), {
        message: /Output port for index [0-9]+ was not found!/,
      });
    });

    it("setOutputTensor(idx, tensor)", () => {
      inferRequest.setOutputTensor(0, resTensor);
      const res2 = inferRequest.getOutputTensor();
      assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
    });

    it("setOutputTensor(idx, object) throws if object is not a Tensor.", () => {
      assert.throws(() => inferRequest.setOutputTensor(0, {}), {
        message: /Argument #[0-9]+ must be a Tensor./,
      });
    });

    it("setOutputTensor() - pass two tensors", () => {
      assert.throws(() => inferRequest.setOutputTensor(resTensor, tensor), {
        message: / invalid argument./,
      });
    });

    it("setTensor(string, tensor)", () => {
      inferRequest.setTensor("fc_out", resTensor);
      const res2 = inferRequest.getTensor("fc_out");
      assert.ok(res2 instanceof ov.Tensor);
      assert.deepStrictEqual(resTensor.data[0], res2.data[0]);
    });

    it("setTensor(string, object) - throws", () => {
      const testName = "testName";
      assert.throws(() => inferRequest.setTensor(testName, tensor), {
        message: /Port for tensor name testName was not found./,
      });
    });

    it("setTensor(string, object) - throws", () => {
      assert.throws(() => inferRequest.setTensor("fc_out", {}), {
        message: /Argument #[0-9]+ must be a Tensor./,
      });
    });

    it("setTensor(string, tensor) - pass one arg", () => {
      assert.throws(() => inferRequest.setTensor("fc_out"), {
        message: / invalid argument./,
      });
    });

    it("setTensor(string, tensor) - pass args in wrong order", () => {
      assert.throws(() => inferRequest.setTensor(resTensor, "fc_out"), {
        message: / invalid argument./,
      });
    });

    it("setTensor(string, tensor) - pass number as first arg", () => {
      assert.throws(() => inferRequest.setTensor(123, "fc_out"), {
        message: / invalid argument/,
      });
    });
  });

  describe("getters", () => {
    let inferRequest = null;
    beforeEach(() => {
      inferRequest = compiledModel.createInferRequest();
      inferRequest.setInputTensor(tensor);
      inferRequest.infer();
    });

    it("Test getTensor(tensorName)", () => {
      const t1 = inferRequest.getTensor("data");
      assert.ok(t1 instanceof ov.Tensor);
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("Test getTensor(Output)", () => {
      const input = inferRequest.getCompiledModel().input();
      const t1 = inferRequest.getTensor(input);
      assert.ok(t1 instanceof ov.Tensor);
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("Test getInputTensor()", () => {
      const t1 = inferRequest.getInputTensor();
      assert.ok(t1 instanceof ov.Tensor);
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("Test getInputTensor(idx)", () => {
      const t1 = inferRequest.getInputTensor(0);
      assert.ok(t1 instanceof ov.Tensor);
      assert.deepStrictEqual(tensor.data[0], t1.data[0]);
    });

    it("Test getOutputTensor(idx?)", () => {
      const res1 = inferRequest.getOutputTensor();
      const res2 = inferRequest.getOutputTensor(0);
      assert.ok(res1 instanceof ov.Tensor);
      assert.ok(res2 instanceof ov.Tensor);
      assert.deepStrictEqual(res1.data[0], res2.data[0]);
    });

    it("Test getCompiledModel()", () => {
      const ir = compiledModel.createInferRequest();
      const cm = ir.getCompiledModel();
      assert.ok(cm instanceof ov.CompiledModel);
      const ir2 = cm.createInferRequest();
      const res2 = ir2.infer([tensorData]);
      const res1 = ir.infer([tensorData]);
      assert.deepStrictEqual(res1["fc_out"].data[0], res2["fc_out"].data[0]);
    });
  });
});

describe("ov.InferRequest tests with missing outputs names", () => {
  const { modelV3Small } = testModels;
  let compiledModel = null;
  let tensorData = null;
  let tensor = null;
  let inferRequest = null;

  before(async () => {
    await isModelAvailable(modelV3Small);

    const core = new ov.Core();

    let modelData = await fs.readFile(modelV3Small.xml, "utf8");
    const weights = await fs.readFile(modelV3Small.bin);
    modelData = modelData.replace('names="MobilenetV3/Predictions/Softmax:0"', "");
    const model = await core.readModel(Buffer.from(modelData, "utf8"), weights);

    compiledModel = await core.compileModel(model, "CPU");
    inferRequest = compiledModel.createInferRequest();

    tensorData = generateImage(modelV3Small.inputShape);
    tensor = new ov.Tensor(ov.element.f32, modelV3Small.inputShape, tensorData);
  });

  it("Test infer(inputData: Tensor[])", () => {
    const result = inferRequest.infer([tensor]);
    assert.deepStrictEqual(Object.keys(result).length, 1);
  });

  it("Test inferAsync(inputData: Tensor[])", () => {
    inferRequest.inferAsync([tensor]).then((result) => {
      assert.deepStrictEqual(Object.keys(result).length, 1);
    });
  });
});
