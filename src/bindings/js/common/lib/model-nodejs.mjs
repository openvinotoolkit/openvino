import { createCanvas, loadImage } from 'canvas';

import Model from "./model.mjs";
import Tensor from "./tensor.mjs";

export default class ModelNodejs extends Model {
  async infer(imgPath, shape) {
    const img = await getImgByPath(imgPath);
    const imgData = await getArrayByImg(img);
    const imgTensor = new Tensor('uint8', imgData, shape);

    return await super.infer(imgTensor);
  }
}

function getImgByPath(path) {
  return loadImage(path);
}

async function getArrayByImg(image) {
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  // Filter alpha channel
  return rgbaData.filter((_, index) => (index + 1)%4);
}
