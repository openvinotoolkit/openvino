
const modelPath = `../assets/models/v3-small_224_1.0_float.xml`;
const imgPath = '../assets/images/coco224x224.jpg';
const shape = [1, 224, 224, 3];
const layout = 'NHWC';

export default async function(openvinojs) {
  const model = await openvinojs.loadModel(modelPath, shape, layout);

  const inputTensor = await getArrayByImgPath(imgPath);

  const outputTensor = await model.infer(inputTensor, shape);

  console.log(outputTensor);
}

const isNodeJS = typeof window === 'undefined';

async function getArrayByImgPath(path) {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = await createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  if (!ctx) throw new Error('Canvas context is null');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;

  return rgbaData.filter((_, index) => (index + 1)%4);
}

async function loadImage(path) {
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

async function createCanvas(width, height) {
  if (isNodeJS) return (await import('canvas')).createCanvas(width, height);

  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
}

async function getClasses() {
  if (isNodeJS) return await import('../assets/imagenet_classes_map.json');

  return fetch('./assets/imagenet_classes_map.json')
    .then((response) => response.json());
}

