
const { default: nodeAddon } = require('openvinojs-node')


const math = require('./lib/helpers.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
const Jimp = require('jimp');


async function onRuntimeInitialized()
{
    const {loadModel, Shape, Tensor,  getDescriptionString} = nodeAddon;
    console.log(await getDescriptionString());
    const model = await loadModel('../assets/models/v3-small_224_1.0_float.xml', '../assets/models/v3-small_224_1.0_float.bin');

    const img_path =  '../assets/images/coco224x224.jpg';
    const jimpSrc = await Jimp.read(img_path);
    const src = cv.matFromImageData(jimpSrc.bitmap);
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);

    //create tensor
    const tensor_data = new Float32Array(src.data);
    const shape = new Shape([1, 224, 224, 3]);
    const tensor = new Tensor("f32", tensor_data, shape);

    /*   ---Perform inference---   */
    const output =  await model.infer(tensor, shape);

    //show the results
    console.log("Result: " + imagenetClassesMap[math.argMax(output.data)]);
    console.log(math.argMax(output.data)); 
}


Module = {
    onRuntimeInitialized
};
cv = require('opencv.js');
