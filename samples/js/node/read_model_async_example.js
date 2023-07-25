const { addon } = require('openvinojs-node');

const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');

run();

async function run()
{
  const { getMaxElement } = await import('../common/index.mjs');
  const imgPath = process.argv[2] || '../assets/images/shih_tzu.jpg';
  const modelPath = '../assets/models/v3-small_224_1.0_float.xml';
  const core = new addon.Core();

  /*   ---Read model asynchronously and create a promise---   */

  const modelPromise = core.readModelAsync(modelPath);

  /*   ---Create a promise with tensor---   */
  const tensorPromise = createTensor(imgPath);

  Promise.all([modelPromise, tensorPromise]).then(([model, tensor]) => {
    const output = model.compile('CPU').infer(tensor);
    //show the results
    const result = getMaxElement(output.data);
    console.log('Result: ' + imagenetClassesMap[result.index],
      '\nIndex: ', result.index);});

}

async function createTensor(imgPath) {
  const { getImageData } = await import('../common/index.mjs');
  const imgData = await getImageData(imgPath);
  const src = cv.matFromImageData(imgData);
  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
  cv.resize(src, src, new cv.Size(224, 224));
  //create tensor
  const tensorData = new Float32Array(src.data);
  const tensor = new addon.Tensor(
    addon.element.f32,
    Int32Array.from([1, 224, 224, 3]),
    tensorData,
  );

  return tensor;
}
