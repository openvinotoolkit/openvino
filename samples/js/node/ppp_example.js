let { addon } = require('openvinojs-node');

const math = require('./lib/helpers.js');
const cv = require('opencv.js');
const imagenetClassesMap = require('../assets/imagenet_classes_map.json');
const Jimp = require('jimp');

run();

async function run()
{

  /*   ---Load the model---   */
  const modelPath = process.argv[2];
  const model = new addon.Model().read_model(modelPath);

  new addon.PrePostProcessor(model)
    .set_input_tensor_shape([1, 224, 224, 3])
    .set_input_tensor_layout('NHWC')
    .set_input_model_layout('NCHW')
    .build();

  /*   ---Load an image---   */
  //read image from a file
  const imgPath = process.argv[3] || '../assets/images/shih_tzu.jpg';
  const jimpSrc = await Jimp.read(imgPath);
  const src = cv.matFromImageData(jimpSrc.bitmap);
  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
  cv.resize(src, src, new cv.Size(224, 224));

  //create tensor
  const tensorData = new Float32Array(src.data);
  math.prepareResnetTensor(tensorData); //Preprocessing needed by resnet network

  const tensor = new addon.Tensor(
    addon.element.f32,
    [1, 224, 224, 3],
    tensorData,
  );

  /*   ---Compile model and perform inference---   */
  const output = model.compile('CPU').infer(tensor);

  //show the results
  console.log('Result: ' + imagenetClassesMap[math.argMax(output.data)]);
  console.log(math.argMax(output.data));
}
