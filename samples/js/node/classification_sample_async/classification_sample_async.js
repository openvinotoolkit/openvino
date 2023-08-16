const { addon: ov } = require('openvinojs-node');

const args = require('args');
const cv2 = require('opencv.js');
const util = require('node:util');
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
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  // (.xml and .bin files) or (.onnx file)
  const model = core.readModel(modelPath);

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imagesData = [];

  for (const imagePath of images)
    imagesData.push(await getImageData(imagePath));

  const tensors = imagesData.map((imgData) => {
    // Use OpenCV.js to preprocess image.
    const originalImage = cv2.matFromImageData(imgData);
    const image = new cv2.Mat();
    // The MobileNet model expects images in RGB format.
    cv2.cvtColor(originalImage, image, cv2.COLOR_RGBA2RGB);
    cv2.resize(image, image, new cv2.Size(224, 224));

    const tensorData = new Float32Array(image.data);
    const shape = [1, image.rows, image.cols, 3];

    return new ov.Tensor(ov.element.f32, shape, tensorData);
  });

  //----------------- Step 4. Apply preprocessing ------------------------------

  new ov.PrePostProcessor(model)
    // FIXME: Uncomment after support tensor in not f32 precision
    // .set_input_element_type(ov.element.u8)
    .setInputTensorLayout('NHWC')
    .setInputModelLayout('NCHW')
    // TODO: add output tensor element type setup
    .build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = core.compileModel(model, deviceName);
  const outputName = compiledModel.output(0).toString();

  //----------------- Step 6. Create infer request queue -----------------------
  console.log('Starting inference in asynchronous mode');

  // Create infer request
  const inferRequest = compiledModel.createInferRequest();

  const promises = tensors.map((t, i) => {
    const promisifiedAsyncInfer = util.promisify(ov.asyncInfer);
    const inferPromise = promisifiedAsyncInfer(inferRequest, [t]);

    inferPromise.then(result =>
      completionCallback(result[outputName], images[i]));

    return inferPromise;
  });

  //----------------- Step 7. Do inference -------------------------------------

  await Promise.all(promises);
  console.log('All inferences executed');

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
