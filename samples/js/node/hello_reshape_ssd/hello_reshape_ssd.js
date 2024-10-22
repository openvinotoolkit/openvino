const { addon: ov } = require('openvino-node');

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
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);

  _ppp.input().tensor()
    .setShape(shape)
    .setElementType(ov.element.u8)
    .setLayout('NHWC');

  _ppp.input().model().setLayout('NCHW');
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
  const output = inferRequest.getTensor(outputLayer);

  const { data: outputData } = output;
  const resultLayer = [];
  const colormap = [[68, 1, 84, 255], [48, 103, 141, 255], [53, 183, 120, 255], [199, 216, 52, 255]];

  const size = outputData.length/4;

  for (let i = 0; i < size; i++) {
    const valueAt = (i, number) => outputData[i + number*size];

    const currentValues = {
      bg: valueAt(i, 0),
      c: valueAt(i, 1),
      h: valueAt(i, 2),
      w: valueAt(i, 3),
    };
    const values = Object.values(currentValues);
    const maxIndex = values.indexOf(Math.max(...values));

    resultLayer.push(maxIndex);
  }

  const pixels = [];
  resultLayer.forEach(i => pixels.push(...colormap[i]));

  const alpha = 0.3;
  const [B, C, H, W] = output.getShape();

  const pixelsAsImageData = arrayToImageData(pixels, W, H);
  const mask = cv.matFromImageData(pixelsAsImageData);

  const originalWidth = image.cols;
  const originalHeight = image.rows;

  cv.resize(mask, mask, new cv.Size(originalWidth, originalHeight));

  cv.addWeighted(mask, alpha, originalImage, 1 - alpha, 0, mask);

  const resultImgData = arrayToImageData(mask.data, originalWidth, originalHeight);
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
