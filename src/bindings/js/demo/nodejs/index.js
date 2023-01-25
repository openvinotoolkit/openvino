const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

const openvinojs = require('../../../../../bin/ia32/Release/openvino_wasm.js');

const FILENAME = 'v3-small_224_1.0_float';

openvinojs().then(async ov => {
  console.log('== start');

  console.log(ov.getVersionString());
  console.log(ov.getDescriptionString());

  // Uploading and creating files on virtual WASM fs
  await uploadFile(`../assets/models/${FILENAME}.bin`, ov);
  await uploadFile(`../assets/models/${FILENAME}.xml`, ov);

  const session = new ov.Session(`${FILENAME}.xml`, `${FILENAME}.bin`);
  const imgData = await getArrayByImgPath('../assets/images/coco.jpg');

  const values = new Uint8Array(imgData);

  let heapSpace = null;
  let heapResult = null;
  try {
    heapSpace = ov._malloc(values.length*values.BYTES_PER_ELEMENT);
    ov.HEAPU8.set(values, heapSpace); 

    heapResult = session.run('[1, 224, 224, 3]', 'NHWC', heapSpace, values.length);
  } finally {
    ov._free(heapSpace);
  }

  const SIZE = 1001;

  const arrayData = [];
  for (let v = 0; v < SIZE; v++) {
    arrayData.push(ov.HEAPF32[heapResult/Float32Array.BYTES_PER_ELEMENT + v]);
  }

  console.log(getMaxArg(arrayData));
  
  console.log('== end');
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

function getMaxArg(arr) {
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
