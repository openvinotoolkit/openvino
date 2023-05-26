const { addon } = require('openvinojs-node');

const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');

run();

async function run()
{
  const { getMaxElement, getImageData } = await import('../common/index.mjs');

  /*   ---Load an image---   */
  const imgPath = process.argv[2] || '../assets/images/shih_tzu.jpg';
  const imgData = await getImageData(imgPath);
  const src = cv.matFromImageData(imgData);
  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
  cv.resize(src, src, new cv.Size(224, 224));
  const tensorData = new Float32Array(src.data);
  const tensor = new addon.Tensor(
    addon.element.f32,
    Int32Array.from([1, 224, 224, 3]),
    tensorData,
  );

  /*   ---Load and compile the model---   */
  const modelPath = '../assets/models/v3-small_224_1.0_float.xml';
  const model = new addon.Model().read_model(modelPath).compile('CPU');

  /*   ---Perform inference---   */
  const output = model.infer(tensor);

  //show the results
  const result = getMaxElement(output.data);
  console.log('Result: ' + imagenetClassesMap[result.index],
    '\nIndex: ', result.index);
}
