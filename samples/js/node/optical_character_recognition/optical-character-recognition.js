const path = require('node:path');
const { addon: ov } = require('openvino-node');

const Image = require('../image.js');
const { transform, argMax, setShape } = require('../helpers.js');

const OUTPUT_PATH = './output/';

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
  // Initialize OpenVINO Core
  const core = new ov.Core();
  // Load the detection model
  const detModel = await core.readModel(detModelXMLPath);
  const detCompiledModel = await core.compileModel(detModel, deviceName);
  const detInputLayer = detCompiledModel.input(0);
  const detOutputLayer = detCompiledModel.output('boxes');

  const img = await Image.load(imagePath);

  // Resize the image to meet network input size
  const [, , detInputHeight, detInputWidth] = detInputLayer.shape;
  const resizedImg = img.resize(detInputWidth, detInputHeight);

  // Prepare input tensor
  const inputImageTransformedData = transform(
    resizedImg.rgb,
    { width: detInputWidth, height: detInputHeight },
    [0, 1, 2],
  );
  const tensorData = new Float32Array(inputImageTransformedData);
  const tensor = new ov.Tensor(ov.element.f32, detInputLayer.shape, tensorData);

  // Run inference on the detection model
  const detInferRequest = detCompiledModel.createInferRequest();
  const detResult = await detInferRequest.inferAsync([tensor]);

  // Load the recognition model
  const recModel = await core.readModel(recModelXMLPath);
  const recModelCompiled = await core.compileModel(recModel, deviceName);
  const recInferRequest = recModelCompiled.createInferRequest();

  // Calculate ratios
  const [ratioX, ratioY] =
    [img.width / detInputWidth, img.height / detInputHeight];
  const boundingBoxesArray = extractBoundingBoxes(detResult[detOutputLayer]);
  // Resize bounding boxes to the original image size
  const boundingBoxesOriginalSizeArray = boundingBoxesArray.map(box =>
    [...multiplyByRatio(ratioX, ratioY, box), box[4]]);

  // Process each bounding box and run inference on the recognition model
  const boxesWithAnnotations = [];
  for (let i = 0; i < boundingBoxesOriginalSizeArray.length; i++) {
    const box = boundingBoxesOriginalSizeArray[i];
    const [xMin, yMin, xMax, yMax] = box;
    const croppedImg = img.crop(xMin, yMin, xMax - xMin, yMax - yMin);
    await croppedImg.save(OUTPUT_PATH + `cropped_image_${i}.jpg`);

    const annotation =
      await performTextRecognition(recModel, recInferRequest, croppedImg);

    boxesWithAnnotations.push({ box, annotation });

    console.log(`Box ${i}: [${box.join(',')}], Annotation: '${annotation}'`);
  }

  const annotatedImg = await putAnnotationsOnTheImage(
    img,
    boxesWithAnnotations,
    { threshold: 0.3, confLabels: false },
  );
  const savePath = path.join(OUTPUT_PATH, 'output_image.jpg');
  await annotatedImg.save(savePath);
  console.log(`The result was saved to ${savePath}`);
}

async function performTextRecognition(model, inferenceRequest, img) {
  const inputLayerShape = model.input(0).shape;
  const outputLayer = model.output(0);

  const [,, inputHeight, inputWidth] = inputLayerShape;
  const resizedImg = img.resize(inputWidth, inputHeight);

  // Convert image to grayscale and create tensor
  const tensor = new ov.Tensor(
    ov.element.f32,
    inputLayerShape,
    new Float32Array(resizedImg.grayscale),
  );

  const result = await inferenceRequest.inferAsync([tensor]);
  const recognitionResults = extractRecognitionResults(result[outputLayer]);
  const annotation = parseAnnotations(recognitionResults);

  return annotation;
}

// Function to extract bounding boxes from the model output
function extractBoundingBoxes(output) {
  const { data: boxes } = output;
  const foldingCoefficient = 5;
  const numberOfBoxes = boxes.length / foldingCoefficient;

  return setShape(boxes, [numberOfBoxes, foldingCoefficient]);
}

// Function to adjust bounding box coordinates by a given ratio
function multiplyByRatio(ratioX, ratioY, box) {
  const scaleShape = (shape, idx) => {
    const position = idx % 2
      ? Math.max(shape * ratioY, 10)
      : shape * ratioX;

    return Math.floor(position);
  }

  return box.map(scaleShape);
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

// Takes original image and bounding boxes with annotations
// and returns the image with annotations
async function putAnnotationsOnTheImage(img, boxesWithAnnotations, options) {
  const defaultOptions = { threshold: 0.3, confLabels: true };
  const { threshold, confLabels } = Object.assign(defaultOptions, options);

  let finalImage = img;

  for (const item of boxesWithAnnotations) {
    const { box, annotation } = item;
    const conf = box[box.length - 1];

    if (conf < threshold) continue;

    const [xMin, yMin, xMax, yMax] = box;
    const yOffset = 10;

    finalImage = finalImage.drawRect(
      xMin, yMin,
      xMax - xMin, yMax - yMin,
      { color: 'green', width: 3 },
    );
    finalImage = finalImage.drawText(
      annotation,
      xMin, yMin - yOffset,
      { font: '30px Arial' },
    );

    if (!confLabels) continue;

    finalImage = finalImage.drawText(
      conf.toFixed(2),
      xMin, yMax + 2 * yOffset,
      { font: '20px Arial' },
    );
  }

  return finalImage;
}
