const { addon: ov } = require('openvino-node');

const Image = require('../image.js');
const imagenetClassesMap = require('../../assets/datasets/imagenet_class_index.json');

// Parsing and validation of input arguments
if (process.argv.length !== 5)
  throw new Error(`Usage: ${process.argv[1]} <path_to_model> `
    + '<path_to_image> <device_name>');

const modelPath = process.argv[2];
const imagePath = process.argv[3];
const deviceName = process.argv[4];

main(modelPath, imagePath, deviceName);

async function main(modelPath, imagePath, deviceName) {
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  const model = await core.readModel(modelPath);

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const img = await Image.load(imagePath);
  const inputTensor = img.toTensor();

  //----------------- Step 4. Apply preprocessing ------------------------------
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().tensor().setElementType(ov.element.u8).setShape(inputTensor.getShape()).setLayout('NHWC');
  _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);
  _ppp.input().model().setLayout('NHWC');
  _ppp.output().tensor().setElementType(ov.element.f32);
  _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, deviceName);

  //---------------- Step 6. Create infer request and do inference synchronously
  console.log('Starting inference in synchronous mode');
  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(inputTensor);
  inferRequest.infer();

  //----------------- Step 7. Process output -----------------------------------
  const outputLayer = compiledModel.outputs[0];
  const resultInfer = inferRequest.getTensor(outputLayer);
  const predictions = Array.from(resultInfer.data)
    .map((prediction, classId) => ({ prediction, classId }))
    .sort(({ prediction: predictionA }, { prediction: predictionB }) =>
      predictionA === predictionB ? 0 : predictionA > predictionB ? -1 : 1);

  const imagenetClasses = ['background', ...Object.values(imagenetClassesMap)];

  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:\n');
  console.log('id\tprobability\tlabel');
  console.log('---------------------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t${prediction.toFixed(7)}\t${imagenetClasses[classId][1]}`),
  );

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
