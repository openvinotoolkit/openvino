export const isNodeJS = typeof window === 'undefined';

/**
 * Read image from a file system.
 * @param {string} path - the path to the image.
 * @returns an ImageData object representing the underlying pixel data
 */
export async function getImageData(path) {
    const image = await loadImage(path);
    const { width, height } = image;
  
    const canvas = await createCanvas(width, height);
    const ctx = canvas.getContext('2d');
  
    if (!ctx) throw new Error('Canvas context is null');
  
    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, width, height);
}


export async function getArrayByImgPath(path) {
  const rgbaData = await getImageData(path);
  return rgbaData.data.filter((_, index) => (index + 1)%4);
}

export async function loadImage(path) {
  if (isNodeJS)
    return (await import('canvas')).loadImage(path);

  return new Promise((resolve) => {
    const img = new Image();

    img.src = path;
    img.onload = () => {
      resolve(img);
    };
  });
}

export async function createCanvas(width, height) {
  if (isNodeJS) return (await import('canvas')).createCanvas(width, height);

  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
}

export function logTensor(tensor) {
  const { value, index } = getMaxElement(tensor.data);

  console.log(`Value: ${value} Index: ${index}`);
}

export async function printClass(tensor) {
  const { index } = getMaxElement(tensor.data);

  const classes = await getClasses();
  console.log(`Class: ${classes[index]}`);
}

export function printShape(shape) {
  console.log(`Tensor shape: ${shape.toString()}`);
}

export async function printOVInfo(openvinojs) {
  console.log(`== OpenVINO v${await openvinojs.getVersionString()}`);
  console.log(`== Description string: ${
    await openvinojs.getDescriptionString()
  }`);
}

async function getClasses() {
  if (isNodeJS)
    return (await import('../assets/imagenet_classes_map.json', {
      assert: { type: 'json' }
    })).default;

  return fetch('./assets/imagenet_classes_map.json')
    .then((response) => response.json());
}

export function getMaxElement(arr) {
  if (!arr.length) return { value: -Infinity, index: -1 };

  let max = arr[0];
  let maxIndex = 0;

  for (let i = 1; i < arr.length; ++i)
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }

  return { value: max, index: maxIndex };
}
