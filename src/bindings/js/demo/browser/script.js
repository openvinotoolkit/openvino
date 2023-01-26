const MODEL_PATH = './assets/models/';
const MODEL_FILENAME = 'v3-small_224_1.0_float';

const statusDiv = document.getElementById('status');

Module().then(async ov => {
  statusDiv.innerText = 'Ready';
  console.log('== start');

  console.log(ov.getVersionString());
  console.log(ov.getDescriptionString());

  // Uploading and creating files on virtual WASM fs
  await uploadFile(`${MODEL_PATH}${MODEL_FILENAME}.bin`, ov);
  await uploadFile(`${MODEL_PATH}${MODEL_FILENAME}.xml`, ov);

  const session = new ov.Session(`${MODEL_FILENAME}.xml`, `${MODEL_FILENAME}.bin`);

  const imgData = await getArrayByImgPath('./assets/images/coco.jpg');

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

  console.log(arrayData);
  console.log(getMaxElement(arrayData));
  
  console.log('== end');
});

async function uploadFile(path, instance) {
  const filename = path.split('/').pop();

  const blob = await fetch(path).then(response => {
    if (!response.ok) {
      return null;
    }     
    return response.blob();
  });

  const buffer = await blob.arrayBuffer();
  const data = new Uint8Array(await blob.arrayBuffer());

  const stream = instance.FS.open(filename, 'w+');
  instance.FS.write(stream, data, 0, data.length, 0);
  instance.FS.close(stream);
}

function loadImage(path) {
  return new Promise((resolve) => {
    const img = new Image();

    img.src = path;
    img.onload = () => {
      resolve(img);
    };
  });
}

function createCanvas(width, height) {
  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
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
