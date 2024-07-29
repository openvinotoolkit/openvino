const { addon: ov } = require('openvino-node');
const fs = require('node:fs');
const path = require('node:path');
const { createCanvas, ImageData } = require('canvas');
const { cv } = require('opencv-wasm');
const {
  transform,
  getImageData,
  argMax,
  setShape,
} = require('../helpers.js');

if (require.main === module) {
// Parsing and validation of input arguments
  if (process.argv.length !== 6)
    throw new Error(
      `Usage: ${process.argv[1]} <path_to_detection_model>` +
      ' <path_to_recognition_model>' +
      ' <path_to_image> <device_name>',
    );

  const detModelXMLPath = process.argv[2];
  const recModelXMLPath = process.argv[3];
  const imagePath = process.argv[4];
  const deviceName = process.argv[5];

  try {
    main(detModelXMLPath, recModelXMLPath, imagePath, deviceName);
  } catch(error) {
    console.error('Error Occurred', error);
  }
}

async function main(detModelXMLPath, recModelXMLPath, imagePath, deviceName) {
  // Initialize OpenVINO core and load the detection mode
  const core = new ov.Core();
  const detModel = await core.readModel(detModelXMLPath);
  const detCompiledModel = await core.compileModel(detModel, deviceName);
  const detInputLayer = detCompiledModel.input(0);
  const detOutputLayer = detCompiledModel.output('boxes');

  const imageData = await getImageData(imagePath);
  const inputImageMat = cv.matFromImageData(imageData);

  // Resize the image to meet network input size
  const [, , H, W] = detInputLayer.shape;
  const resizedImage = new cv.Mat();
  cv.cvtColor(inputImageMat, inputImageMat, cv.COLOR_RGBA2RGB);
  cv.cvtColor(inputImageMat, inputImageMat, cv.COLOR_BGR2RGB);
  cv.resize(inputImageMat, resizedImage, new cv.Size(W, H));

  // Prepare input tensor
  const inputImage = transform(
    resizedImage.data,
    { width: W, height: H },
    [0, 1, 2],
  );
  const tensorData = new Float32Array(inputImage);
  const tensor = new ov.Tensor(ov.element.f32, detInputLayer.shape, tensorData);

  const detInferRequest = detCompiledModel.createInferRequest();

  const detResult = await detInferRequest.inferAsync([tensor]);
  const boundingBoxesArray = extractBoundingBoxes(detResult[detOutputLayer]);

  const recModel = await core.readModel(recModelXMLPath);
  const recModelCompiled = await core.compileModel(recModel, deviceName);
  const recInputLayer = recModelCompiled.input(0);
  const recOutputLayer = recModelCompiled.output(0);

  // Process each bounding box and run inference on the recognition model
  const [, , height, width] = recInputLayer.shape;
  // Calculate ratios
  const { ratioX, ratioY } = calculateRatios(inputImageMat, resizedImage);

  // Convert image to grayscale
  const grayscaleImage = convertToGrayscale(inputImageMat);

  const annotations = [];
  const croppedImages = [];

  for (let i = 0; i < boundingBoxesArray.length; i++) {
    const crop = boundingBoxesArray[i];
    const [xMin, yMin, xMax, yMax] = multiplyByRatio(ratioX, ratioY, crop).map(
      Math.floor,
    );
    const cropRect = new cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin);
    const croppedImage = grayscaleImage.roi(cropRect);

    try {
      const preprocessedCrop = resizeAndConvertCropToModelInput(croppedImage, [
        width,
        height,
      ]);
      const tensorData = new Float32Array(preprocessedCrop);
      const tensor = new ov.Tensor(
        ov.element.f32,
        Int32Array.from(recInputLayer.shape),
        tensorData,
      );

      await inferAsyncProcess(
        tensor,
        recModelCompiled,
        recOutputLayer,
        i,
        annotations,
      );

      croppedImages.push(cropImage(inputImageMat, xMin, yMin, xMax, yMax));
    } catch(error) {
      console.error('Error during preprocessing:', error);
    }

    croppedImage.delete();
  }

  grayscaleImage.delete();

  const boxesWithAnnotations = boundingBoxesArray.map((box, index) => ({
    box,
    annotation: annotations[index],
  }));

  logBoxesWithAnnotations(boxesWithAnnotations);

  convertResultToImage(
    inputImageMat,
    resizedImage,
    boxesWithAnnotations,
    { threshold: 0.3, confLabels: true },
    './assets/results/output_image.jpg',
  );

  croppedImages.forEach((croppedImage, i) => {
    const savePath = `./assets/results/cropped_image_${i}.jpg`;
    saveImage(croppedImage, savePath);
  });
}

// Function to extract bounding boxes from the model output
function extractBoundingBoxes(output) {
  const { data: boxes } = output;
  const foldingCoefficient = 5;
  const numberOfBoxes = boxes.length / foldingCoefficient;

  return setShape(boxes, [numberOfBoxes, foldingCoefficient]);
}

// Function to calculate the ratios for the image
function calculateRatios(originalImage, resizedImage) {
  const realY = originalImage.rows;
  const realX = originalImage.cols;
  const resizedY = resizedImage.rows;
  const resizedX = resizedImage.cols;
  const ratioX = realX / resizedX;
  const ratioY = realY / resizedY;

  return { ratioX, ratioY };
}

// Function to convert the image to grayscale
function convertToGrayscale(originalImage) {
  const grayscaleImage = new cv.Mat();
  cv.cvtColor(originalImage, grayscaleImage, cv.COLOR_BGR2GRAY);

  return grayscaleImage;
}

// Function to adjust bounding box coordinates by a given ratio
function multiplyByRatio(ratioX, ratioY, box) {
  const scaleShape = (shape, idx) =>
    idx % 2 ? Math.max(shape * ratioY, 10) : shape * ratioX;

  return box.map(scaleShape);
}

// Function to resize and convert a crop to the recognition model input format
function resizeAndConvertCropToModelInput(crop, netShape) {
  const [netWidth, netHeight] = netShape;

  // Resize the crop to the network's input shape
  const tempImg = new cv.Mat();
  cv.resize(crop, tempImg, new cv.Size(netWidth, netHeight));

  // Create the reshaped buffer
  const reshapedBuffer = new Uint8Array(netHeight * netWidth);
  let index = 0;

  for (let i = 0; i < netHeight; i++) {
    for (let j = 0; j < netWidth; j++) {
      reshapedBuffer[index++] = tempImg.ucharPtr(i, j)[0];
    }
  }

  // Clean up
  tempImg.delete();

  return reshapedBuffer;
}

// Function to extract recognition results from the model output
function extractRecognitionResults(output) {
  const outputData = output.getData();
  const outputShape = output.getShape();
  const [, height, width] = outputShape;

  return setShape(outputData, [height, width]);
}

// Function to parse annotations from the recognition results
function parseAnnotations(recognitionResults) {
  const letters = '~0123456789abcdefghijklmnopqrstuvwxyz';
  const annotation = [];

  for (const row of recognitionResults) {
    const letterIndex = argMax(row);
    const parsedLetter = letters[letterIndex];

    // Stop if end character is encountered
    if (parsedLetter === letters[0]) break;
    annotation.push(parsedLetter);
  }

  return annotation.join('');
}

// Function to crop the image based on the bounding box coordinates
function cropImage(originalImage, xMin, yMin, xMax, yMax) {
  xMin = Math.max(0, xMin);
  yMin = Math.max(0, yMin);
  xMax = Math.min(originalImage.cols, xMax);
  yMax = Math.min(originalImage.rows, yMax);
  if (xMin >= xMax || yMin >= yMax) {
    throw new Error('Invalid crop coordinates');
  }
  const roi = originalImage.roi(
    new cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin),
  );
  const cropped = new cv.Mat();
  roi.copyTo(cropped);
  roi.delete();

  return cropped;
}

// Get Text size
function getTextSize(text, fontFace, fontScale) {
  const canvas = createCanvas(200, 200);
  const ctx = canvas.getContext('2d');
  const adjustedFontScale = fontScale * 35;
  ctx.font = `${adjustedFontScale}px ${fontFace}`;
  const metrics = ctx.measureText(text);
  const width = metrics.width;
  const height =
    metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;

  return { width, height };
}

/* The convertResultToImage function visualizes object detection
 results on an image by drawing bounding boxes around detected
 objects and optionally adding labels to them. */

function convertResultToImage(
  bgrImage,
  resizedImage,
  boxesWithAnnotations,
  options,
  savePath,
) {
  const defaultOptions = { threshold: 0.3, confLabels: true };
  const { threshold, confLabels } = Object.assign(defaultOptions, options);

  const colors = {
    red: [255, 0, 0, 255],
    green: [0, 255, 0, 255],
    white: [255, 255, 255, 255],
  };
  const [realY, realX] = [bgrImage.rows, bgrImage.cols];
  const [resizedY, resizedX] = [resizedImage.rows, resizedImage.cols];
  const [ratioX, ratioY] = [realX / resizedX, realY / resizedY];

  const rgbImage = new cv.Mat();
  cv.cvtColor(bgrImage, rgbImage, cv.COLOR_BGR2RGB);

  boxesWithAnnotations.forEach(({ box, annotation }) => {
    const conf = box[box.length - 1];

    if (conf < threshold) return;

    const [xMin, yMin, xMax, yMax] = multiplyByRatio(ratioX, ratioY, box);

    cv.rectangle(
      rgbImage,
      new cv.Point(xMin, yMin),
      new cv.Point(xMax, yMax),
      colors.green,
      3,
    );

    if (!confLabels) return;

    const text = `${annotation}`;
    const fontScale = 0.8;
    const thickness = 1;
    const { width: textW, height: textH } = getTextSize(
      text,
      'Arial',
      fontScale,
    );
    const imageCopy = rgbImage.clone();

    cv.rectangle(
      imageCopy,
      new cv.Point(xMin, yMin - textH - 10),
      new cv.Point(xMin + textW, yMin - 10),
      colors.white,
      cv.FILLED,
    );
    cv.addWeighted(imageCopy, 0.4, rgbImage, 0.6, 0, rgbImage);
    cv.putText(
      rgbImage,
      text,
      new cv.Point(xMin, yMin - 10),
      cv.FONT_HERSHEY_SIMPLEX,
      fontScale,
      colors.red,
      thickness,
      cv.LINE_AA,
    );

    imageCopy.delete();
  });

  const saveDir = path.dirname(savePath);
  if (!fs.existsSync(saveDir)) {
    fs.mkdirSync(saveDir, { recursive: true });
  }

  try {
    saveImage(rgbImage, savePath);
  } catch(e) {
    console.log(`Error occurred while saving ----> ${e}`);
  }

  return rgbImage;
}

// Infer async helper function

async function inferAsyncProcess(
  tensor,
  recModelCompiled,
  recOutputLayer,
  i,
  annotations,
) {
  // Create infer request
  const inferRequest = recModelCompiled.createInferRequest();

  // Define the completion callback function
  function completionCallback(outputTensor, i, annotations) {
    const recognitionResults = extractRecognitionResults(outputTensor);
    const annotation = parseAnnotations(recognitionResults);
    annotations.push(annotation);
  }

  // Start inference in asynchronous mode
  try {
    const result = await inferRequest.inferAsync([tensor]);
    completionCallback(result[recOutputLayer], i, annotations);
  } catch(error) {
    console.error('Error during inference:', error);
  }
}

// Log boudning boxes with annotations
function logBoxesWithAnnotations(boxesWithAnnotations) {
  boxesWithAnnotations.forEach((item, i) => {
    const { box, annotation } = item;
    console.log(`Box ${i}: [${box}], Annotation: ${annotation}`);
  });
}

function saveImage(rgbImage, savePath) {
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
    for (let i = 0; i < rgbImage.data.length; i++) {
      if (i % 3 === 0) imgDataArr.push(255);
      imgDataArr.push(rgbImage.data[i]);
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
  if (!fs.existsSync(saveDir)) {
    fs.mkdirSync(saveDir, { recursive: true });
  }

  fs.writeFileSync(savePath, imageBuffer);
  console.log('Image saved successfully!', savePath);
}
