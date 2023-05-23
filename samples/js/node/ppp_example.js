const { addon } = require('openvinojs-node');

const math = require('./lib/helpers.js');
const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
const Jimp = require('jimp');

run();

async function run()
{

  /*   ---Load an image---   */
  //read image from a file
  const imgPath = process.argv[3] || '../assets/images/shih_tzu.jpg';
  const jimpSrc = await Jimp.read(imgPath);
  const imgSource = cv.matFromImageData(jimpSrc.bitmap);
  cv.cvtColor(imgSource, imgSource, cv.COLOR_RGBA2BGR);
  cv.resize(imgSource, imgSource, new cv.Size(227, 227));

  /*   ---Load the model---   */
  const modelPath = process.argv[2];
  const model = new addon.Model().read_model(modelPath);

  new addon.PrePostProcessor(model)
    .set_input_tensor_shape([1, 227, 227, 3])
    .set_input_tensor_layout('NHWC')
    .set_input_model_layout('NCHW')
    .build();

  const tensor = new addon.Tensor(
    addon.element.f32,
    [1, 227, 227, 3],
    new Float32Array(imgSource.data),
  );

  /*   ---Compile model and perform inference---   */
  const output = model.compile('CPU').infer(tensor);

  //show the results
  console.log('Result: ' + imagenetClassesMap[math.argMax(output.data)]);
  console.log(math.argMax(output.data));
}
