const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

const openvinojs = require('../dist/openvino_wasm.js');

const MODEL_PATH = '../assets/models/';
const MODEL_FILENAME = 'v3-small_224_1.0_float';
const IMAGE_PATH = '../assets/images/coco.jpg';

openvinojs().then(async ov => {
  console.log('= Start');

  const imagenetClassesMap = (await import('../assets/imagenet_classes_map.mjs')).default;

  console.log(`== OpenVINO v${ov.getVersionString()}`);
  console.log(`== Description string: ${ov.getDescriptionString()}`);

  // Uploading and creating files on virtual WASM fs
  await uploadFile(`${MODEL_PATH}${MODEL_FILENAME}.bin`, ov);
  await uploadFile(`${MODEL_PATH}${MODEL_FILENAME}.xml`, ov);

  const session = new ov.Session(`${MODEL_FILENAME}.xml`, `${MODEL_FILENAME}.bin`, '[1, 224, 224, 3]', 'NHWC');
  const imgData = await getArrayByImgPath(IMAGE_PATH);

  const values = new Uint8Array(imgData);

  let heapSpace = null;
  let heapResult = null;
  try {
    heapSpace = ov._malloc(values.length*values.BYTES_PER_ELEMENT);
    ov.HEAPU8.set(values, heapSpace); 

    console.time('== Inference time');
    heapResult = session.run(heapSpace, values.length);
    console.timeEnd('== Inference time');
  } finally {
    ov._free(heapSpace);
  }

  const SIZE = session.outputTensorSize;

  const arrayData = [];
  for (let v = 0; v < SIZE; v++) {
    arrayData.push(ov.HEAPF32[heapResult/Float32Array.BYTES_PER_ELEMENT + v]);
  }

  console.log('== Output vector:');
  console.log(arrayData);

  const max = getMaxElement(arrayData);
  console.log(`== Max index: ${max.index}, value: ${max.value}`);
  console.log(`== Result class: ${imagenetClassesMap[max.index]}`);
  
  console.log('= End');
});

async function uploadFile(path, ov) {
  const filename = path.split('/').pop();
  const fileData = fs.readFileSync(path);

  const data = new Uint8Array(fileData);

  const stream = ov.FS.open(filename, 'w+');
  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}

async function getArrayByImgPath(path) {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  return rgbaData.filter((_, index) => (index + 1)%4);
}

function getMaxElement(arr) {
  if (!arr.length) return { value: -Infinity, index: -1 };

  let max = arr[0];
  let maxIndex = 0;

  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }

  return { value: max, index: maxIndex };
}
