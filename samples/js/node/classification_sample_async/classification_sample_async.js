const { addon: ov } = require('openvinojs-node');

const args = require('args');
const { cv } = require('opencv-wasm');
const { getImageData } = require('../helpers.js');

args.options([{
  name: 'img',
  defaultValue: [],
}, {
  name: 'model',
}, {
  name: 'device',
}]);
const { model: modelPath, device: deviceName, img: images } =
  args.parse(process.argv);

main(modelPath, images, deviceName);

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

async function main(modelPath, images, deviceName) {
  //----------- Step 1. Initialize OpenVINO Runtime Core -----------------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------- Step 2. Read a model -------------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  // (.xml and .bin files) or (.onnx file)
  const model = await core.readModel(modelPath);
  const [h, w] = model.inputs[0].shape.slice(-2);
  const tensorShape = [1, h, w, 3];

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------- Step 3. Set up input -------------------------------------------
  // Read input image
  const imagesData = [];

  for (const imagePath of images)
    imagesData.push(await getImageData(imagePath));

  const preprocessedImages = imagesData.map((imgData) => {
    // Use opencv-wasm to preprocess image.
    const originalImage = cv.matFromImageData(imgData);
    const image = new cv.Mat();
    // The MobileNet model expects images in RGB format.
    cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);
    cv.resize(image, image, new cv.Size(w, h));

    return new Uint8Array(image.data);
  });

  //----------- Step 4. Apply preprocessing ------------------------------------
  const _ppp = new ov.PrePostProcessor(model);
  _ppp.input().tensor().setLayout('NHWC').setElementType(ov.element.u8);
  _ppp.input().model().setLayout('NCHW');
  _ppp.build();
  // TODO: add output tensor element type setup


  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, deviceName);
  const outputName = compiledModel.output(0).toString();

  //----------- Step 6. Collecting promises to react when they resolve ---------
  console.log('Starting inference in asynchronous mode');

  // Create infer request
  const inferRequest = compiledModel.createInferRequest();

  const promises = preprocessedImages.map((tensorData, i) => {
    const inferPromise = inferRequest.inferAsync([new ov.Tensor(ov.element.u8, tensorShape, tensorData)]);

    inferPromise.then(result =>
      completionCallback(result[outputName], images[i]));

    return inferPromise;
  });

  //----------- Step 7. Do inference -------------------------------------------
  await Promise.all(promises);
  console.log('All inferences executed');

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
