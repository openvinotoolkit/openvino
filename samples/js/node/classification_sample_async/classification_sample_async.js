const { addon: ov } = require('openvino-node');

const args = require('args');
const Image = require('../image.js');

args.options([
  {
    name: 'img',
    defaultValue: [],
  },
  {
    name: 'model',
  },
  {
    name: 'device',
  },
]);
const {
  model: modelPath,
  device: deviceName,
  img: imgPaths
} = args.parse(process.argv);

main(modelPath, imgPaths, deviceName);

function completionCallback(result, imagePath) {
  const predictions = Array.from(result.data)
    .map((prediction, classId) => ({ prediction, classId }))
    .sort(({ prediction: predictionA }, { prediction: predictionB }) =>
      predictionA === predictionB ? 0 : predictionA > predictionB ? -1 : 1);

  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}`),
  );
  console.log();
}

async function main(modelPath, imgPaths, deviceName) {
  //----------- Step 1. Initialize OpenVINO Runtime Core -----------------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------- Step 2. Read a model -------------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  // (.xml and .bin files) or (.onnx file)
  const model = await core.readModel(modelPath);

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------- Step 3. Set up input -------------------------------------------
  const inputTensors = [];
  const [_, w, h] = model.inputs[0].getShape();

  // Read input image, resize it to the model's input size and convert it to a tensor.
  for (const path of imgPaths) {
    const img = await Image.load(path);
    const resized = img.resize(w, h);

    inputTensors.push(resized.toTensor());
  }

  //----------- Step 4. Apply preprocessing ------------------------------------
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().tensor().setLayout('NHWC').setElementType(ov.element.u8);
  _ppp.input().model().setLayout('NHWC');
  _ppp.output().tensor().setElementType(ov.element.f32);
  _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, deviceName);
  const outputName = compiledModel.output(0).toString();

  //----------- Step 6. Collecting promises to react when they resolve ---------
  console.log('Starting inference in asynchronous mode');

  // Create infer request
  const inferRequest = compiledModel.createInferRequest();
  const promises = inputTensors.map((tensor, i) => {
    const inferPromise = inferRequest.inferAsync([tensor]);

    inferPromise.then(result =>
      completionCallback(result[outputName], imgPaths[i]));

    return inferPromise;
  });

  //----------- Step 7. Do inference -------------------------------------------
  await Promise.all(promises);
  console.log('All inferences executed');

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
