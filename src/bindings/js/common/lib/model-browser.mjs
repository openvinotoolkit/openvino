import Model from "./model.mjs";
import Tensor from "./tensor.mjs";

export default class ModelBrowser extends Model {
  async infer(imgPathOrImgDataArray, shape) {
    const imgData = typeof imgPathOrImgDataArray === 'string' 
      ? await getArrayByImgPath(imgPathOrImgDataArray)
      : imgPathOrImgDataArray;
    const imgTensor = new Tensor('uint8', imgData, shape);

    return await super.infer(imgTensor);
  }
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
