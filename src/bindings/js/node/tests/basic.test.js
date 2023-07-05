const ov = require('../build/Release/ov_node_addon.node');
const path = require('path');

function getModelPath(is_fp16=false){
    basePath = "../../python/tests/"
    if (is_fp16){
        test_xml = path.join(basePath, "test_utils", "utils", "test_model_fp16.xml")
        test_bin = path.join(basePath, "test_utils", "utils", "test_model_fp16.bin")
    } else {
        test_xml = path.join(basePath, "test_utils", "utils", "test_model_fp32.xml")
        test_bin = path.join(basePath, "test_utils", "utils", "test_model_fp32.bin")
    }
    return (test_xml, test_bin)
}



var test_xml, test_bin = getModelPath();


describe('Output class for ov::Output<ov::Node> ', () => {
    
    const core = new ov.Core();
    const model = core.read_model(test_xml);

    test('Model.output() method', () => {
        // TO_DO check if object is an instance of a value/class
        expect(model.output() && typeof model.output() === 'object').toBe(true)
    });

    test('Model.outputs property', () => {
        expect(model.outputs.length).toEqual(1);     
    });

    test('Model.output().ToString() method', () => {
        //test for a model with one output
        expect(model.output().toString()).toEqual("fc_out");
    });

    test('Model.output(idx: number).ToString() method', () => {
        expect(model.output(0).toString()).toEqual("fc_out");
    });

    test('Model.output(tensorName: string).ToString() method', () => {
        expect(model.output("fc_out").toString()).toEqual("fc_out");
    });

    test('Ouput.shape property with dimensions', () => {
        expect(model.output(0).shape).toEqual([1, 10]);
    });

    test('Ouput.getShape() method', () => {
        expect(model.output(0).getShape().getData()).toEqual([1, 10]);
    });

});


describe('Output class for ov::Output<const ov::Node>', () => {
    const core = new ov.Core();
    const model = core.read_model(test_xml);
    const compiledModel = core.compile_model(model, 'CPU');

    test('CompiledModel.output() method', () => {
        // TO_DO check if object is an instance of a value/class
        expect(compiledModel.output() && typeof compiledModel.output() === 'object').toBe(true)
    });

    test('CompiledModel.outputs property', () => {
        expect(compiledModel.outputs.length).toEqual(1);     
    });

    test('CompiledModel.output().ToString() method', () => {
        //test for a model with one output
        expect(compiledModel.output().toString()).toEqual("fc_out");
    });

    test('CompiledModel.output(idx: number).ToString() method', () => {
        expect(compiledModel.output(0).toString()).toEqual("fc_out");
    });

    test('CompiledModel.output(tensorName: string).ToString() method', () => {
        expect(compiledModel.output("fc_out").toString()).toEqual("fc_out");
    });

    test('Ouput.shape property with dimensions', () => {
        expect(compiledModel.output(0).shape).toEqual([1, 10]);
    });

    test('Ouput.getShape() method', () => {
        expect(compiledModel.output(0).getShape().getData()).toEqual([1, 10]);
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
