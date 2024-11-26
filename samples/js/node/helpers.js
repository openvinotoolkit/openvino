const path = require('node:path');
const { createWriteStream } = require('node:fs');
const { mkdir, stat } = require('node:fs/promises');
const { HttpsProxyAgent } = require('https-proxy-agent');

module.exports = {
  exp,
  sum,
  triu,
  tril,
  argMax,
  reshape,
  getShape,
  setShape,
  transform,
  downloadFile,
  extractValues,
  matrixMultiplication,
};

function transform(arr, { width, height }, order) {
  // Calculate the number of pixels and the size of each channel
  const numPixels = width * height;
  const channels = [[], [], []];

  // Separate RGB channels
  for (let i = 0; i < numPixels; i++) {
    channels[0].push(arr[i * 3]);     // Red channel
    channels[1].push(arr[i * 3 + 1]); // Green channel
    channels[2].push(arr[i * 3 + 2]); // Blue channel
  }

  // Reorder channels based on the 'order' array
  const reorderedChannels = order.map(num => channels[num]);

  // Flatten reordered channels into a single array
  const result = reorderedChannels.flat();

  return result;
}

async function downloadFile(url, filename, destination) {
  const { env } = process;
  const timeout = 5000;

  await applyFolderPath(destination);

  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);
  const protocolString = new URL(url).protocol === 'https:' ? 'https' : 'http';
  const module = require(`node:${protocolString}`);
  const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  let agent;

  if (proxyUrl) {
    agent = new HttpsProxyAgent(proxyUrl);
    console.log(`Proxy agent configured using: '${proxyUrl}'`);
  }

  return new Promise((resolve, reject) => {
    file.on('error', e => {
      reject(`Error oppening file stream: ${e}`);
    });

    const getRequest = module.get(url, { agent }, res => {
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
      const innerArray = createMultidimensionArray(flatArray, remainingShape,
        offset + i*innerArrayLength);

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
    throw new Error('Number of columns in the first matrix must match the '
      + 'number of rows in the second matrix.');

  const result = [];

  for (let i = 0; i < rows1; i++) {
    result[i] = [];

    for (let j = 0; j < cols2; j++) {
      let sum = 0;

      for (let k = 0; k < cols1; k++)
        sum += matrix1[i][k] * matrix2[k][j];

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

    for (let j = 0; j < numCols; j++)
      result[i][j] = i <= j - k ? matrix[i][j] : 0;
  }

  return result;
}

function tril(matrix, k = 0) {
  const numRows = matrix.length;
  const numCols = matrix[0].length;
  const result = [];

  for (let i = 0; i < numRows; i++) {
    result[i] = [];

    for (let j = 0; j < numCols; j++)
      result[i][j] = i >= j - k ? matrix[i][j] : 0;
  }

  return result;
}

async function applyFolderPath(dirPath) {
  try {
    await stat(dirPath);

    return;
  } catch(err) {
    if (err.code !== 'ENOENT') throw err;

    await mkdir(dirPath, { recursive: true });
  }
}
