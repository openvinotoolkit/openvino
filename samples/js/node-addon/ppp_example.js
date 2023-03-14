var ov = require('bindings')('ov_node_addon.node');


const math = require('./lib/math_func.js');
const Jimp = require('jimp');
const fs = require('fs');
const imagenet_classes = fs.readFileSync('./imagenet_2012_labels.txt').toString().split("\n");

async function onRuntimeInitialized()
{

    
    /*   ---Load the model---   */
    const model_path = process.argv[3];
    const model = new ov.Model().read_model(model_path)

    ppp = new ov.PrePostProcessor(model)
                    .set_input_tensor_shape([1, 224, 224, 3])
                    .set_input_tensor_layout("NHWC")
                    .set_input_model_layout("NCHW")
                    .build();
    


    /*   ---Load an image---   */
    //read image from a file
    const img_path = process.argv[2];
    const jimpSrc = await Jimp.read(img_path);
    const src = cv.matFromImageData(jimpSrc.bitmap);
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
    cv.resize(src, src, new cv.Size(224, 224));

    
    //create tensor
    const tensor_data = new Float32Array(src.data);
    math.prepare_resnet_tensor(tensor_data); //Preprocessing needed by resnet network

    const tensor = new ov.Tensor(ov.element.f32, [1, 224, 224, 3], tensor_data);

    /*   ---Compile model and perform inference---   */
    const output = model.compile("CPU").infer(tensor);

    //show the results
    console.log("Result: " + imagenet_classes[math.argMax(output.data) - 1]);
}



Module = {
    onRuntimeInitialized
};
cv = require('./lib/opencv.js');
