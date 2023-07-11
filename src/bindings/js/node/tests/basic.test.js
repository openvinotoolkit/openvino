const ov = require('../build/Release/ov_node_addon.node');
const path = require('path');

function getModelPath(isFP16=false){
    basePath = "../../python/tests/"
    if (isFP16){
        test_xml = path.join(basePath, "test_utils", "utils", "test_model_fp16.xml")
        test_bin = path.join(basePath, "test_utils", "utils", "test_model_fp16.bin")
    } else {
        test_xml = path.join(basePath, "test_utils", "utils", "test_model_fp32.xml")
        test_bin = path.join(basePath, "test_utils", "utils", "test_model_fp32.bin")
    }
    return (test_xml, test_bin)
}



var test_xml, test_bin = getModelPath();


describe('Output class', () => {
    
    const core = new ov.Core();
    const model = core.read_model(test_xml);
    const compiledModel = core.compile_model(model, 'CPU');

    it.each([
        [model],
        [compiledModel]
    ])('Mutual tests', (obj) => {
        expect(obj.output() && typeof obj.output() === 'object').toBe(true);
        expect(obj.outputs.length).toEqual(1);  
        //test for a obj with one output
        expect(obj.output().toString()).toEqual("fc_out");
        expect(obj.output(0).toString()).toEqual("fc_out");
        expect(obj.output("fc_out").toString()).toEqual("fc_out");
        expect(obj.output(0).shape).toEqual([1, 10]);
        expect(obj.output(0).getShape().getData()).toEqual([1, 10]);
        expect(obj.output().getAnyName()).toEqual("fc_out");
    });

    test('Ouput<ov::Node>.setNames() method', () => {
        model.output().setNames(["bTestName", "cTestName"]);
        expect(model.output().getAnyName()).toEqual("bTestName");
    });

    test('Ouput<ov::Node>.addNames() method', () => {
        model.output().addNames(["aTestName"]);
        expect(model.output().getAnyName()).toEqual("aTestName");
    });

    test('Ouput<const ov::Node>.setNames() method', () => {
        expect(() => compiledModel.output().setNames(["bTestName", "cTestName"])).toThrow(TypeError);
    });

    test('Ouput<const ov::Node>.addNames() method', () => {
        expect(() => compiledModel.output().addNames(["aTestName"])).toThrow(TypeError);
    });

});

describe('Input class for ov::Input<const ov::Node>', () => {
    const core = new ov.Core();
    const model = core.read_model(test_xml);
    const compiledModel = core.compile_model(model, 'CPU');

    test('CompiledModel.input() method', () => {
        // TO_DO check if object is an instance of a value/class
        expect(compiledModel.input() && typeof compiledModel.input() === 'object').toBe(true)
    });

    test('CompiledModel.inputs property', () => {
        expect(compiledModel.inputs.length).toEqual(1);     
    });

    test('CompiledModel.input().ToString() method', () => {
        //test for a model with one output
        expect(compiledModel.input().toString()).toEqual("data");
    });

    test('CompiledModel.input(idx: number).ToString() method', () => {
        expect(compiledModel.input(0).toString()).toEqual("data");
    });

    test('CompiledModel.input(tensorName: string).ToString() method', () => {
        expect(compiledModel.input("data").toString()).toEqual("data");
    });

    test('Input.shape property with dimensions', () => {
        expect(compiledModel.input(0).shape).toEqual([1, 3, 32, 32]);
    });

    test('Input.getShape() method', () => {
        expect(compiledModel.input(0).getShape().getData()).toEqual([1, 3, 32, 32]);
    });
});
