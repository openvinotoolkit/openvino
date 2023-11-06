const { addon: ov } = require('openvinojs-node');

const fs = require('node:fs/promises');
const { cv } = require('opencv-wasm');
const {
  setShape,
  getImageData,
  getImageBuffer,
  arrayToImageData,
} = require('../helpers.js');

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
  // (.xml and .bin files) or (.onnx file)
  const model = await core.readModel(modelPath);

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imgData = await getImageData(imagePath);

  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);

  const tensorData = new Uint8Array(image.data);
  const shape = [1, image.rows, image.cols, 3];
  const inputTensor = new ov.Tensor(ov.element.u8, shape, tensorData);

  //----------------- Step 4. Apply preprocessing ------------------------------
  const _ppp = new ov.PrePostProcessor(model);
  _ppp.input().preprocess().resize(ov.resizeAlgorithm.RESIZE_LINEAR);

  _ppp.input().tensor()
    .setShape(shape)
    .setElementType(ov.element.u8)
    .setLayout('NHWC');

  // TODO: add output tensor element type setup
  _ppp.input().model().setLayout('NCHW');
  _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, deviceName);

  //---------------- Step 6. Create infer request and do inference synchronously
  console.log('Starting inference in synchronous mode');
  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(inputTensor);
  inferRequest.inferSync();

  //----------------- Step 7. Process output -----------------------------------
  const outputLayer = compiledModel.outputs[0];
  const resultInfer = inferRequest.getTensor(outputLayer);
  const predictions = Array.from(resultInfer.data);
  const [height, width] = [originalImage.rows, originalImage.cols];

  const detections = setShape(predictions, [100, 7]);
  const color = [255, 0, 0, 255];
  const THROUGHPUT = 0.9;

  detections.forEach(detection => {
    const [classId, confidence, xmin, ymin, xmax, ymax] = detection.slice(1);

    if (confidence < THROUGHPUT) return;

    console.log(`Found: classId = ${classId}, `
      + `confidence = ${confidence.toFixed(2)}, `
      + `coords = (${xmin}, ${ymin}), (${xmax}, ${ymax})`,
    );

    // Draw a bounding box on a output image
    cv.rectangle(originalImage,
      new cv.Point(xmin*width, ymin*height),
      new cv.Point(xmax*width, ymax*height),
      color,
      2,
    );
  });

  const resultImgData = arrayToImageData(originalImage.data, width, height);
  const filename = 'out.jpg';

  await fs.writeFile(`./${filename}`, getImageBuffer(resultImgData));

  try {
    await fs.readFile(filename);
    console.log('Image out.jpg was created!');
  } catch(err) {
    console.log(`Image ${filename} was not created. Check your permissions.`);
  }

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
