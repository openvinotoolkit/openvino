const { default: nodeAddon } = require('openvinojs-node');

const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');

run();

async function run()
{
  const { getMaxElement, getImageData } = await import('../common/index.mjs');
  const {loadModel, Shape, Tensor, getDescriptionString} = nodeAddon;
  console.log(await getDescriptionString());
  const model = await loadModel(
    '../assets/models/v3-small_224_1.0_float.xml',
    '../assets/models/v3-small_224_1.0_float.bin',
  );

  const imgPath = '../assets/images/coco224x224.jpg';
  const imgData = await getImageData(imgPath);
  const src = cv.matFromImageData(imgData);
  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);

  //create tensor
  const tensorData = new Float32Array(src.data);
  const shape = new Shape([1, 224, 224, 3]);
  const tensor = new Tensor('f32', tensorData, shape);

  /*   ---Perform inference---   */
  const output = await model.infer(tensor, shape);

  //show the results
  const result = getMaxElement(output.data);
  console.log('Result: ' + imagenetClassesMap[result.index],
    '\nIndex: ', result.index);
}
