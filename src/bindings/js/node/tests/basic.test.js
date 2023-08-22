const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const path = require('path');

function getModelPath(isFP16=false) {
  const basePath = '../../python/tests/';
  if (isFP16) {
    testXml = path.join(basePath, 'test_utils', 'utils', 'test_model_fp16.xml');
  } else {
    testXml = path.join(basePath, 'test_utils', 'utils', 'test_model_fp32.xml');
  }

  return testXml;
}

var testXml = getModelPath();
const core = new ov.Core();
const model = core.readModel(testXml);
const compiledModel = core.compileModel(model, 'CPU');

describe('Output class', () => {
  const modelLike = [[model],
    [compiledModel]];

  modelLike.forEach( ([obj]) => {
    it('Output getters and properties', () => {
      assert.strictEqual(typeof obj.output(), 'object');
      assert.strictEqual(obj.outputs.length, 1);
      // tests for an obj with one output
      assert.strictEqual(obj.output().toString(), 'fc_out');
      assert.strictEqual(obj.output(0).toString(), 'fc_out');
      assert.strictEqual(obj.output('fc_out').toString(), 'fc_out');
      assert.deepStrictEqual(obj.output(0).shape, [1, 10]);
      assert.deepStrictEqual(obj.output(0).getShape(), [1, 10]);
      assert.strictEqual(obj.output().getAnyName(), 'fc_out');
      assert.strictEqual(obj.output().anyName, 'fc_out');
    });
  });

  it('Ouput<ov::Node>.setNames() method', () => {
    model.output().setNames(['bTestName', 'cTestName']);
    assert.strictEqual(model.output().getAnyName(), 'bTestName');
    assert.strictEqual(model.output().anyName, 'bTestName');
  });

  it('Ouput<ov::Node>.addNames() method', () => {
    model.output().addNames(['aTestName']);
    assert.strictEqual(model.output().getAnyName(), 'aTestName');
    assert.strictEqual(model.output().anyName, 'aTestName');
  });

  it('Ouput<const ov::Node>.setNames() method', () => {
    assert.throws(
      () => compiledModel.output().setNames(['bTestName', 'cTestName'])
    );
  });

  it('Ouput<const ov::Node>.addNames() method', () => {
    assert.throws(
      () => compiledModel.output().addNames(['aTestName']),
    );
  });

});

describe('Input class for ov::Input<const ov::Node>', () => {
  it('CompiledModel.input() method', () => {
    // TO_DO check if object is an instance of a value/class
    assert.strictEqual(typeof compiledModel.input(), 'object');
  });

  it('CompiledModel.inputs property', () => {
    assert.equal(compiledModel.inputs.length, 1);
  });

  it('CompiledModel.input().ToString() method', () => {
    //test for a model with one output
    assert.strictEqual(compiledModel.input().toString(), 'data');
  });

  it('CompiledModel.input(idx: number).ToString() method', () => {
    assert.strictEqual(compiledModel.input(0).toString(), 'data');
  });

  it('CompiledModel.input(tensorName: string).ToString() method', () => {
    assert.strictEqual(compiledModel.input('data').toString(), 'data');
  });

  it('Input.shape property with dimensions', () => {
    assert.deepStrictEqual(compiledModel.input(0).shape, [1, 3, 32, 32]);
  });

  it('Input.getShape() method', () => {
    assert.deepStrictEqual(
      compiledModel.input(0).getShape(), [1, 3, 32, 32]);
  });
});

describe('InferRequest infer()', () => {
  const inferRequest = compiledModel.createInferRequest();
  const tensorData = Float32Array.from({ length: 3072 }, () => Math.random());
  const tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 32, 32]),
    tensorData,
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

  tensorLike.forEach(([tl]) => {
    inferRequest.infer([tl]);
    const result = inferRequest.getOutputTensor();
    const label = tl instanceof Float32Array ? 'TypedArray' : 'Tensor';
    it(`Test getOutputTensor(): ${label} ])`, () => {
      assert.deepStrictEqual(result.data.length, 10);
    });
  });

  it('Test setInputTensor()', () => {
    inferRequest.setInputTensor(tensor);
    inferRequest.infer();
    assert.deepStrictEqual(inferRequest.getOutputTensor().data.length, 10);
  });

    // Segmentation fault
//   it('Test failed setInputTensor()', () => {
//     assert.throws(
//       () => inferRequest.setInputTensor(tensorData),
//       /Passed value is not a valid tensor./
//     );
//   });

});
