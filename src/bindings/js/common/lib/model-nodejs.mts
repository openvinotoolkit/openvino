import { createCanvas, loadImage } from 'canvas';

import Model from './model.mjs';
import Tensor from './tensor.mjs';

import type { Image } from 'canvas';
import type { IShape, ITensor } from './types.mjs';

export default class ModelNodejs extends Model {
  // @ts-ignore: FIXME: Align signatures
  async infer(imgPath: string, shape: IShape): ITensor {
    const img = await getImgByPath(imgPath);
    const imgData = await getArrayByImg(img);
    const imgTensor = new Tensor('uint8', imgData, shape);

    return await super.infer(imgTensor);
  }
}

function getImgByPath(path: string): Promise<Image> {
  return loadImage(path);
}

async function getArrayByImg(image: Image): Promise<Uint8ClampedArray> {
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  // @ts-ignore: FIXME: Align Image & HTMLImageElement
  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  // Filter alpha channel
  return rgbaData.filter((_, index) => (index + 1)%4);
}
