const { addon } = require('openvinojs-node');

const math = require('./lib/helpers.js');
const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
const Jimp = require('jimp');

run();

async function run()
{
  const imgPath = process.argv[2] || '../assets/images/shih_tzu.jpg';
  const modelPath = '../assets/models/v3-small_224_1.0_float.xml';
  const core = new addon.Core();

  /*   ---Read model asynchronously and create a promise---   */

  const modelPromise = core.read_model_async(modelPath);

  /*   ---Create a promise with tensor---   */
  const tensorPromise = createTensor(imgPath);

  Promise.all([modelPromise, tensorPromise]).then(([model, tensor]) => {
    const output = model.compile('CPU').infer(tensor);
    //show the results
    console.log('Result: ' + imagenetClassesMap[math.argMax(output.data)]);
    console.log(math.argMax(output.data));
  });

}

async function createTensor(imgPath) {
  const jimpSrc = await Jimp.read(imgPath);
  const src = cv.matFromImageData(jimpSrc.bitmap);
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
