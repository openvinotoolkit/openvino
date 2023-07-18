const fs = require('fs');
const http = require('http');
const https = require('https');
const { createCanvas, createImageData, loadImage, Image, ImageData } = require('canvas');

exports.arrayToImageData = arrayToImageData;
exports.displayImage = displayImage;
exports.getImageData = getImageData;
exports.displayArrayAsImage = displayArrayAsImage;
exports.transform = transform;
exports.downloadFile = downloadFile;
exports.extractValues = extractValues;
exports.setShape = setShape;
exports.exp = exp;

function arrayToImageData(array, width, height) {
  return createImageData(new Uint8ClampedArray(array), width, height);
}

function displayImage(imageOrImageData, display) {
  const canvas = createCanvas(imageOrImageData.width, imageOrImageData.height);
  const ctx = canvas.getContext('2d');

  if (imageOrImageData instanceof Image)
    ctx.drawImage(imageOrImageData, 0, 0);
  else if (imageOrImageData instanceof ImageData)
    ctx.putImageData(imageOrImageData, 0, 0);
  else
    throw Error(`Passed parameters has type '${typeof imageOrImageData}'. It is't supported.`);

  const buffer = canvas.toBuffer('image/jpeg');

  display.image(buffer);
}

function displayArrayAsImage(arr, width, height, display) {
  const alpha = 255;
  const componentsPerPixel = arr.length/(width*height);

  try {
    switch(componentsPerPixel) {
      case 1:
        arr = arr.reduce((acc, val) => {
          acc.push(val, val, val, alpha);

          return acc;
        }, []);
        break;
      case 3:
        arr = arr.reduce((acc, val, index) => {
          if (index && index%3 === 0) acc.push(alpha);

          acc.push(val);

          return acc;
        }, []);
        break;
    }
  } catch(e) {
    console.log(e);
  }

  const imageData = arrayToImageData(arr, width, height);

  displayImage(imageData, display);
}

async function getImageData(path) {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = await createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);

  return ctx.getImageData(0, 0, width, height);
}

function transform(arr, { width, height }, order) {
  const img = new cv2.Mat(height, width, cv2.CV_8UC3);

  img.data.set(arr, 0, arr.length);

  const channels = new cv2.MatVector();
  cv2.split(img, channels);

  const val = order.map(num => [...channels.get(num).data]);

  return [].concat(...val);
}

function downloadFile(url, filename, destination) {
  const timeout = 5000;
  const fullPath = destination + '/' + filename;
  const file = fs.createWriteStream(fullPath);
  const module = new URL(url).protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    file.on('error', e => {
      reject(`Error oppening file stream: ${e}`);
    });

    const getRequest = module.get(url, res => {
      const { statusCode } = res;

      if (statusCode !== 200)
        return reject(`Server returns status code: ${statusCode}`);

      res.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`File successfully stored at '${fullPath}'`);
        resolve();
      });
    });

    getRequest.on('error', e => {
      reject(`Error sending request: ${e}`);
    });

    getRequest.setTimeout(timeout, () => {
      getRequest.destroy();
      reject(`Request timed out after ${timeout}`);
    });
  });
}

function sum(array) {
  return array.reduce((acc, val) => acc+val, 0);
}

function mul(array) {
  return array.reduce((acc, val) => acc*val, 1);
}

function setShape(flatArray, shape) {
  if (mul(shape) !== flatArray.length)
    throw new Error('Shape doesn\'t according to array length');

  return createMultidimensionArray(flatArray, shape, 0);
}

function createMultidimensionArray(flatArray, shape, offset) {
  const currentDim = shape[0];
  const remainingShape = shape.slice(1);
  const currentArray = [];

  if (remainingShape.length === 0) {
    for (let i = 0; i < currentDim; i++)
      currentArray.push(flatArray[offset + i]);
  }
  else {
    const innerArrayLength = mul(shape) / currentDim;

    for (let i = 0; i < currentDim; i++) {
      const innerArray = createMultidimensionArray(flatArray, remainingShape, offset + i*innerArrayLength);
      currentArray.push(innerArray);
    }
  }

  return currentArray;
}

function extractValues(arrOrVal, collector = []) {
  if (arrOrVal[Symbol.iterator] && arrOrVal.map) {
    arrOrVal.map(v => extractValues(v, collector));
  }
  else {
    collector.push(arrOrVal);
  }

  return collector;
}

function isIterableArray(arr) {
  return arr[Symbol.iterator] && arr.map;
}

function eachInner(arrOrValue, fn) {
  return isIterableArray(arrOrValue)
    ? arrOrValue.map(e => eachInner(e, fn))
    : fn(arrOrValue);
}

function exp(arr) {
  return eachInner(arr, Math.exp);
}

function sumRows(arr) {
  return arr.map(row => sum(row));
}

function reshape(arr, newShape) {
  const flat = extractValues(arr);

  return setShape(flat, newShape);
}

function getShape(arr, acc = []) {
  if (isIterableArray(arr)) {
    acc.push(arr.length);
    getShape(arr[0], acc);
  }

  return acc;
}

function matrixMultiplication(matrix1, matrix2) {
  const rows1 = matrix1.length;
  const cols1 = matrix1[0].length;
  const rows2 = matrix2.length;
  const cols2 = matrix2[0].length;

  if (cols1 !== rows2)
    throw new Error('Number of columns in the first matrix must match the number of rows in the second matrix.');

  const result = [];

  for (let i = 0; i < rows1; i++) {
    result[i] = [];
    for (let j = 0; j < cols2; j++) {
      let sum = 0;
      for (let k = 0; k < cols1; k++) {
        sum += matrix1[i][k] * matrix2[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}

function findMax(arr) {
  let max = -Infinity;
  let index = -1;

  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < max) continue;

    max = arr[i];
    index = i;
  }

  return { value: max, index };
}

function argMax(arr) {
  return findMax(arr).index;
}


function triu(matrix, k = 0) {
  const numRows = matrix.length;
  const numCols = matrix[0].length;

  const result = [];

  for (let i = 0; i < numRows; i++) {
    result[i] = [];
    for (let j = 0; j < numCols; j++) {
      if (i <= j - k) {
        result[i][j] = matrix[i][j];
      } else {
        result[i][j] = 0;
      }
    }
  }

  return result;
}

function tril(matrix, k = 0) {
  const numRows = matrix.length;
  const numCols = matrix[0].length;

  const result = [];

  for (let i = 0; i < numRows; i++) {
    result[i] = [];
    for (let j = 0; j < numCols; j++) {
      if (i >= j - k) {
        result[i][j] = matrix[i][j];
      } else {
        result[i][j] = 0;
      }
    }
  }

  return result;
}

function arange(count) {
  const arr = new Array(count);

  for (let i = 0; i < count; i++) arr[i] = i;

  return arr;
}

exports.exp = exp;
exports.sum = sum;
exports.sumRows = sumRows;
exports.reshape = reshape;
exports.getShape = getShape;
exports.argMax = argMax;
exports.triu = triu;
exports.tril = tril;
exports.arange = arange;
exports.matrixMultiplication = matrixMultiplication;
