const { addon: ov } = require('openvino-node');
const Image = require('../image.js');

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
  const img = await Image.load(imagePath);
  const inputTensor = img.toTensor();

  //----------------- Step 4. Apply preprocessing ------------------------------
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);

  _ppp.input().tensor()
    .setShape(inputTensor.getShape())
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
  const outputs = inferRequest.infer([inputTensor]);

  //----------------- Step 7. Process output -----------------------------------
  const outputLayer = compiledModel.outputs[0];
  const output = outputs[outputLayer];
  const outputData = output.data;
  const resultLayer = [];
  const colormap = [
    [68, 1, 84, 255],
    [48, 103, 141, 255],
    [53, 183, 120, 255],
    [199, 216, 52, 255],
  ];
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

  const alpha = 0.6;
  const filename = 'out.jpg';
  const [, , H, W] = output.getShape();

  const segmentsImg = Image.fromArray(pixels, W, H);
  const resizedSegments = segmentsImg.resize(img.width, img.height);
  const mergedImg = Image.overlay(img, resizedSegments, alpha);

  try {
    await mergedImg.save(filename);
    console.log(`Image '${filename}' was created.`);
  } catch(err) {
    console.log(`Image '${filename}' was not created. Check your permissions.`);
  }

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}
