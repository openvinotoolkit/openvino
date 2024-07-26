const { cv } = require('opencv-wasm');
const fs = require('node:fs').promises;
const path = require('node:path');
const { addon: ov } = require('openvino-node');
const { createCanvas, ImageData } = require('canvas');
const { getImageData, transform, setShape } = require('../helpers');

if (require.main === module) {
// Parsing and validation of input arguments
  if (process.argv.length !== 6)
    throw new Error(
      `Usage: ${process.argv[1]} <path_to_unet_model>` +
      ' <path_to_foreground_image>' +
      ' <path_to_background_image> <device_name>',
    );

  const unetModelPath = process.argv[2];
  const foreGroundImage = process.argv[3];
  const backGroundImage = process.argv[4];
  const deviceName = process.argv[5];

  try {
    main(unetModelPath, foreGroundImage, backGroundImage, deviceName);
  } catch(error) {
    console.error('Error occurred', error);
  }
}

async function main(
  unetModelPath,
  foreGroundImage,
  backGroundImage,
  deviceName,
) {
  const core = new ov.Core();

  // Read and compile model
  const model = await core.readModel(unetModelPath);
  const compiledModel = await core.compileModel(model, deviceName);

  // Get the names of input and output layers.
  const inputLayer = compiledModel.input(0);
  const outputLayer = compiledModel.output(0);

  // Get Image data from the foreground image
  const imageData = await getImageData(foreGroundImage);
  const inputImageMat = cv.matFromImageData(imageData);

  //  Convert the image shape to a shape and a data type expected by the network
  const [, , H, W] = inputLayer.shape;
  const resizedImage = new cv.Mat();
  cv.cvtColor(inputImageMat, inputImageMat, cv.COLOR_BGR2RGB);
  cv.resize(inputImageMat, resizedImage, new cv.Size(W, H));

  const inputImage = transform(
    resizedImage.data,
    { width: W, height: H },
    [0, 1, 2],
  );

  // Normalize the input image Mat
  const normalizedInputImage = normalizeImage(inputImage, W, H);

  // Create a tensor from the normalized input image
  const tensorData = new Float32Array(normalizedInputImage);
  const tensor = new ov.Tensor(ov.element.f32, inputLayer.shape, tensorData);

  // Do inference
  const inferRequest = compiledModel.createInferRequest();
  const inferResult = await inferRequest.inferAsync([tensor]);

  const { data } = inferResult[outputLayer];
  const reshapedResult = setShape(data, [512, 512]);

  // Create a Mat from the reshaped result
  const reshapedMat = cv.matFromArray(
    512,
    512,
    cv.CV_32F,
    reshapedResult.flat(),
  );

  // Get the height and width of the original image
  const height = inputImageMat.rows;
  const width = inputImageMat.cols;

  // Resize the inference result to the original image size
  const resizedResult = new cv.Mat();
  cv.resize(
    reshapedMat,
    resizedResult,
    new cv.Size(width, height),
    0,
    0,
    cv.INTER_LINEAR,
  );

  // Convert the resized result to uint8
  resizedResult.convertTo(resizedResult, cv.CV_8U);

  // Create a Mat to store the background removed result
  const bgRemovedResult = inputImageMat.clone();

  removeBackground(resizedResult, bgRemovedResult);

  // Save the background removed result
  await saveImage(bgRemovedResult, './bg_removed_result.jpg');

  // Get the background image data
  const bgrImageData = await getImageData(backGroundImage);
  const bgrImageMat = cv.matFromImageData(bgrImageData);

  // Resize the background image to the original image size
  const resizedBgrImageMat = new cv.Mat();
  cv.cvtColor(bgrImageMat, bgrImageMat, cv.COLOR_BGR2RGB);
  cv.resize(bgrImageMat, resizedBgrImageMat, new cv.Size(width, height));

  // Remove the foreground from the background image by
  // setting all foreground pixels to white
  removeForeground(resizedResult, resizedBgrImageMat);

  // Save the foreground removed from the background image
  await saveImage(resizedBgrImageMat, './fg_removed_from_background.jpg');

  // create a new Mat to store the final image
  const newImage = new cv.Mat(
    resizedBgrImageMat.rows,
    resizedBgrImageMat.cols,
    cv.CV_8UC3,
  );

  // combine the foreground and background images to get the final image
  combineImages(resizedResult, bgRemovedResult, resizedBgrImageMat, newImage);

  // Save the final image
  await saveImage(newImage, './background_changed_image.jpg');
}

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

function removeBackground(mask, image) {
  // Iterate over the mask and set all background pixels to white
  for (let i = 0; i < mask.rows; i++) {
    for (let j = 0; j < mask.cols; j++) {
      if (mask.ucharPtr(i, j)[0] === 0) {
        image.ucharPtr(i, j)[0] = 255;
        image.ucharPtr(i, j)[1] = 255;
        image.ucharPtr(i, j)[2] = 255;
      }
    }
  }
}

function removeForeground(mask, image) {
  // Iterate over the mask and set all foreground pixels to black
  for (let i = 0; i < mask.rows; i++) {
    for (let j = 0; j < mask.cols; j++) {
      if (mask.ucharPtr(i, j)[0] === 1) {
        image.ucharPtr(i, j)[0] = 0;
        image.ucharPtr(i, j)[1] = 0;
        image.ucharPtr(i, j)[2] = 0;
      } else {
        image.ucharPtr(i, j)[0] = image.ucharPtr(i, j)[0];
        image.ucharPtr(i, j)[1] = image.ucharPtr(i, j)[1];
        image.ucharPtr(i, j)[2] = image.ucharPtr(i, j)[2];
      }
    }
  }
}

function combineImages(mask, fgImage, bgImage, newImage) {
  // Iterate over the mask and combine the foreground and background images
  for (let i = 0; i < mask.rows; i++) {
    for (let j = 0; j < mask.cols; j++) {
      if (mask.ucharPtr(i, j)[0] === 1) {
        newImage.ucharPtr(i, j)[0] = fgImage.ucharPtr(i, j)[0];
        newImage.ucharPtr(i, j)[1] = fgImage.ucharPtr(i, j)[1];
        newImage.ucharPtr(i, j)[2] = fgImage.ucharPtr(i, j)[2];
      } else {
        newImage.ucharPtr(i, j)[0] = bgImage.ucharPtr(i, j)[0];
        newImage.ucharPtr(i, j)[1] = bgImage.ucharPtr(i, j)[1];
        newImage.ucharPtr(i, j)[2] = bgImage.ucharPtr(i, j)[2];
      }
    }
  }
}

async function saveImage(rgbImage, savePath) {
  const canvas = createCanvas(rgbImage.cols, rgbImage.rows);
  const ctx = canvas.getContext('2d');
  const componentsPerPixel =
    rgbImage.data.length / (rgbImage.cols * rgbImage.rows);
  const imgDataArr = [];

  if (componentsPerPixel === 1) {
    for (const val of rgbImage.data) {
      imgDataArr.push(val, val, val, 255);
    }
  } else if (componentsPerPixel === 3) {
    for (let i = 0; i < rgbImage.data.length; i += 3) {
      imgDataArr.push(
        rgbImage.data[i + 2], // Red
        rgbImage.data[i + 1], // Green
        rgbImage.data[i], // Blue
        255, // Alpha
      );
    }
  }

  const imageData = new ImageData(
    new Uint8ClampedArray(imgDataArr),
    rgbImage.cols,
    rgbImage.rows,
  );
  ctx.putImageData(imageData, 0, 0);

  const dataURL = canvas.toDataURL('image/jpeg');
  const base64Data = dataURL.replace(/^data:image\/jpeg;base64,/, '');
  const imageBuffer = Buffer.from(base64Data, 'base64');

  const saveDir = path.dirname(savePath);
  try {
    await fs.mkdir(saveDir, { recursive: true });
    await fs.writeFile(savePath, imageBuffer);
    console.log('Image saved successfully!', savePath);
  } catch(error) {
    console.error('Error saving image:', error);
  }
}
