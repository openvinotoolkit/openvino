var { addon } = require('openvinojs-node');


const math = require('./lib/math_func.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
const Jimp = require('jimp');
const fs = require('fs');



async function onRuntimeInitialized()
{

    /*   ---Load an image---   */
    //read image from a file
    const img_path = process.argv[2] || '../assets/images/shih_tzu.jpg';
    const jimpSrc = await Jimp.read(img_path);
    const src = cv.matFromImageData(jimpSrc.bitmap);
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
    cv.resize(src, src, new cv.Size(224, 224));
    //create tensor
    const tensor_data = new Float32Array(src.data);
    const tensor = new addon.Tensor(addon.element.f32, Int32Array.from([1, 224, 224, 3]), tensor_data);


    /*   ---Load and compile the model---   */
    const model_path = '../assets/models/v3-small_224_1.0_float.xml';
    model = new addon.Model().read_model(model_path).compile("CPU");

    /*   ---Perform inference---   */
    const output = model.infer(tensor);

    //show the results
    console.log("Result: " + imagenetClassesMap[math.argMax(output.data)]);
    console.log(math.argMax(output.data));
}


Module = {

    onRuntimeInitialized
};
cv = require('opencv.js');

