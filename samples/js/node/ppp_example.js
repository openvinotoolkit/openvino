var { addon } = require('openvinojs-node');


const math = require('./lib/math_func.js');
const Jimp = require('jimp');
const fs = require('fs');

async function onRuntimeInitialized()
{


    /*   ---Load the model---   */
    const model_path = process.argv[2];
    const model = new addon.Model().read_model(model_path)

    ppp = new addon.PrePostProcessor(model)
                    .set_input_tensor_shape([1, 224, 224, 3])
                    .set_input_tensor_layout("NHWC")
                    .set_input_model_layout("NCHW")
                    .build();



    /*   ---Load an image---   */
    //read image from a file
    const img_path = process.argv[3] || '../assets/images/shih_tzu.jpg';
    const jimpSrc = await Jimp.read(img_path);
    const src = cv.matFromImageData(jimpSrc.bitmap);
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
    cv.resize(src, src, new cv.Size(224, 224));


    //create tensor
    const tensor_data = new Float32Array(src.data);
    math.prepare_resnet_tensor(tensor_data); //Preprocessing needed by resnet network

    const tensor = new addon.Tensor(addon.element.f32, [1, 224, 224, 3], tensor_data);

    /*   ---Compile model and perform inference---   */
    const output = model.compile("CPU").infer(tensor);

    //show the results
    const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
    console.log("Result: " + imagenetClassesMap[math.argMax(output.data)]);
}



Module = {
    onRuntimeInitialized
};
cv = require('./lib/opencv.js');
