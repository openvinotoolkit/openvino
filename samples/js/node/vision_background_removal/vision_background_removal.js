const { addon: ov } = require('openvino-node');

const { transform } = require('../helpers');
const Image = require('../image');

if (require.main === module) {
  // Parsing and validation of input arguments
  if (process.argv.length !== 6)
    throw new Error(
      `Usage: ${process.argv[1]} <path_to_unet_model>` +
      ' <path_to_foreground_image>' +
      ' <path_to_background_image> <device_name>',
    );

  const unetModelPath = process.argv[2];
  const foregroundImagePath = process.argv[3];
  const backgroundImagePath = process.argv[4];
  const deviceName = process.argv[5];

  try {
    main(unetModelPath, foregroundImagePath, backgroundImagePath, deviceName);
  } catch(error) {
    console.error('Error occurred', error);
  }
}

module.exports = main;

async function main(
  unetModelPath,
  foregroundImagePath,
  backgroundImagePath,
  deviceName,
) {
  const core = new ov.Core();

  // Read and compile model
  const model = await core.readModel(unetModelPath);
  const compiledModel = await core.compileModel(model, deviceName);

  // Get the names of input and output layers
  const inputLayer = compiledModel.input(0);
  const outputLayer = compiledModel.output(0);

  // Load foreground image
  const originalImg = await Image.load(foregroundImagePath);

  // Resize image to a shape expected by the network
  const [, , modelInputHeight, modelInputWidth] = inputLayer.shape;
  const resized = await originalImg.resize(modelInputWidth, modelInputHeight);

  // Create a tensor from the normalized input image
  const transformed = transform(
    resized.rgb,
    {
      width: modelInputWidth,
      height: modelInputHeight
    },
    [0, 1, 2]
  );
  const normalizedInputImage = normalizeImage(
    transformed,
    modelInputWidth,
    modelInputHeight,
  );
  const tensor = new ov.Tensor(ov.element.f32, inputLayer.shape, normalizedInputImage);

  // Do inference
  const inferRequest = compiledModel.createInferRequest();
  const inferResult = await inferRequest.inferAsync([tensor]);
  const { data: resultData } = inferResult[outputLayer];

  // Normalize the result data from grayscale to RGB
  const rgbData = [];
  for (let i = 0; i < resultData.length; i += 1) {
    const value = resultData[i] * 255;

    rgbData.push(value, value, value, 255);
  }

  // Create image based on result data
  const [outputHeight, outputWidth] = outputLayer.shape.slice(2);
  const maskImg = await Image.fromArray(rgbData, outputWidth, outputHeight);

  // Resize the result mask to the original image size and save it
  const { width, height } = originalImg;
  const resizedMaskImg = await maskImg.resize(originalImg.width, originalImg.height);
  const maskImagePath = './out_mask.jpg';
  await resizedMaskImg.save(maskImagePath);
  console.log(`The mask image was saved to '${maskImagePath}'`);

  // Remove the foreground from the original image
  const removedBgImg = Image.mask(originalImg, resizedMaskImg);

  // Load the background image
  const bgrImage = await Image.load(backgroundImagePath);

  // Resize the background image to the same size as the original image
  const resizedBgrImage = bgrImage.resize(width, height);

  // Remove object from the background image
  const removedFgImg = Image.mask(resizedBgrImage, resizedMaskImg.invert());

  // Combine the background and foreground images
  const resultImg = Image.merge(removedBgImg, removedFgImg);

  // Save the final image
  const outputImagePath = './out_bgr_changed_image.jpg';
  await resultImg.save(outputImagePath);
  console.log(`The result image was saved to '${outputImagePath}'`);
  console.log('The background was successfully changed');
}

// Details about this normalization:
// https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vision-background-removal/vision-background-removal.ipynb#Load-and-Pre-Process-Input-Image
function normalizeImage(imageData, width, height) {
  // Mean and scale values
  const inputMean = [123.675, 116.28, 103.53];
  const inputScale = [58.395, 57.12, 57.375];

  const normalizedData = new Float32Array(imageData.length);
  const channels = 3;

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      for (let c = 0; c < channels; c++) {
        const index = i * width * channels + j * channels + c;

        normalizedData[index] =
          (imageData[index] - inputMean[c]) / inputScale[c];
      }
    }
  }

  return normalizedData;
}
