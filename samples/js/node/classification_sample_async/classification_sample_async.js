const { addon: ov } = require('openvinojs-node');

const args = require('args');
const cv2 = require('opencv.js');
const { getImageData } = require('../helpers.js');

args.options([{
  name: 'img',
  defaultValue: [],
}, {
  name: 'model',
}, {
  name: 'device',
}]);
const parsedArgs = args.parse(process.argv);

// Parsing and validation of input arguments
// if (process.argv.length !== 5)
//   throw new Error(`Usage: ${process.argv[1]} <path_to_model> `
//     + '<path_to_image> <device_name>');

const { model: modelPath, device: deviceName, images } = parsedArgs;

main(modelPath, images, deviceName);

function completionCallback(inferRequest, imagePath) {
  const [resultInfer] = inferRequest.getTensors();
  const predictions = Array.from(resultInfer.data);

  // TODO: add sorting by probability

  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}`),
  );
  console.log();
}

async function main(modelPath, imagePath, deviceName) {
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

  for (const imagePath in images)
    imagesData.push(await getImageData(imagePath));

  const tensors = imagesData.map((imgData) => {
    // Use OpenCV.js to preprocess image.
    const originalImage = cv2.matFromImageData(imgData);
    const image = new cv2.Mat();
    // The MobileNet model expects images in RGB format.
    cv2.cvtColor(originalImage, image, cv2.COLOR_RGBA2RGB);

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

  //----------------- Step 6. Create infer request queue -----------------------
  console.log('Starting inference in asynchronous mode');

  // create async queue with optimal number of infer requests
  const inferQueue = ov.AsyncInferQueue(compiledModel);
  inferQueue.setCallback(completionCallback);

  //----------------- Step 7. Do inference -------------------------------------

  const inferencePromises = tensors.map((t, i) =>
    inferQueue.startAsync({ 0: t }, images[i]));

  await Promise.all(inferencePromises);
  console.log('All inferences executed');

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
