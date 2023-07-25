const ov = require('../build/Release/ov_node_addon.node');
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

describe('Output class', () => {

  const core = new ov.Core();
  const model = core.readModel(testXml);
  const compiledModel = core.compileModel(model, 'CPU');

  test.each([
    model,
    compiledModel,
  ])('Mutual tests', (obj) => {
    expect(obj.output() && typeof obj.output() === 'object').toBe(true);
    expect(obj.outputs.length).toEqual(1);
    //test for a obj with one output
    expect(obj.output().toString()).toEqual('fc_out');
    expect(obj.output(0).toString()).toEqual('fc_out');
    expect(obj.output('fc_out').toString()).toEqual('fc_out');
    expect(obj.output(0).shape).toEqual([1, 10]);
    expect(obj.output(0).getShape().getData()).toEqual([1, 10]);
    expect(obj.output().getAnyName()).toEqual('fc_out');
    expect(obj.output().anyName).toEqual('fc_out');
  });

  test('Ouput<ov::Node>.setNames() method', () => {
    model.output().setNames(['bTestName', 'cTestName']);
    expect(model.output().getAnyName()).toEqual('bTestName');
    expect(model.output().anyName).toEqual('bTestName');
  });

  test('Ouput<ov::Node>.addNames() method', () => {
    model.output().addNames(['aTestName']);
    expect(model.output().getAnyName()).toEqual('aTestName');
    expect(model.output().anyName).toEqual('aTestName');
  });

  test('Ouput<const ov::Node>.setNames() method', () => {
    expect(() => compiledModel.output().setNames(['bTestName', 'cTestName'])).toThrow(TypeError);
  });

  test('Ouput<const ov::Node>.addNames() method', () => {
    expect(() => compiledModel.output().addNames(['aTestName'])).toThrow(TypeError);
  });

});

describe('Input class for ov::Input<const ov::Node>', () => {
  const core = new ov.Core();
  const model = core.readModel(testXml);
  const compiledModel = core.compileModel(model, 'CPU');

  test('CompiledModel.input() method', () => {
    // TO_DO check if object is an instance of a value/class
    expect(compiledModel.input() && typeof compiledModel.input() === 'object').toBe(true);
  });

  test('CompiledModel.inputs property', () => {
    expect(compiledModel.inputs.length).toEqual(1);
  });

  test('CompiledModel.input().ToString() method', () => {
    //test for a model with one output
    expect(compiledModel.input().toString()).toEqual('data');
  });

  test('CompiledModel.input(idx: number).ToString() method', () => {
    expect(compiledModel.input(0).toString()).toEqual('data');
  });

  test('CompiledModel.input(tensorName: string).ToString() method', () => {
    expect(compiledModel.input('data').toString()).toEqual('data');
  });

  test('Input.shape property with dimensions', () => {
    expect(compiledModel.input(0).shape).toEqual([1, 3, 32, 32]);
  });

  test('Input.getShape() method', () => {
    expect(compiledModel.input(0).getShape().getData()).toEqual([1, 3, 32, 32]);
  });
});

describe('InferRequest infer()', () => {
  const core = new ov.Core();
  const model = core.readModel(testXml);
  const compiledModel = core.compileModel(model, 'CPU');
  tensor_data = Float32Array.from({ length: 3072 }, () => Math.random());

  const tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 32, 32]),
    tensor_data,
  );

  test.each([
    tensor,
    tensor_data,
  ])('Different tensor-like values', (val) => {
    const inferRequest = compiledModel.createInferRequest();
    inferRequest.infer({ data: val });
    const result = inferRequest.getOutputTensors();
    expect(Object.keys(result)).toEqual(['fc_out']);
    expect(result['fc_out'].data.length).toEqual(10);
  });

  test.each([
    tensor,
    tensor_data,
  ])('Different tensor-like values', (val) => {
    const inferRequest2 = compiledModel.createInferRequest();
    inferRequest2.infer([val]);
    const result2 = inferRequest2.getOutputTensors();
    expect(Object.keys(result2)).toEqual(['fc_out']);
    expect(result2['fc_out'].data.length).toEqual(10);
  });

});
