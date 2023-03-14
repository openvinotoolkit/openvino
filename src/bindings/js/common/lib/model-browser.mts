import Model from './model.mjs';
import Tensor from './tensor.mjs';

import type { IShape, ITensor } from './types.mjs';

export default class ModelBrowser extends Model {
  // @ts-ignore: FIXME: Align signatures
  async infer(imgPathOrImgDataArray: string | number[], shape: IShape): Promise<ITensor> {
    const imgData = typeof imgPathOrImgDataArray === 'string' 
      ? await getArrayByImgPath(imgPathOrImgDataArray)
      : imgPathOrImgDataArray;
    const imgTensor = new Tensor('uint8', imgData, shape);

    return await super.infer(imgTensor);
  }
}

async function getArrayByImgPath(path: string): Promise<Uint8ClampedArray> {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  if (!ctx) throw new Error('Canvas context is null');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  return rgbaData.filter((_, index) => (index + 1)%4);
}

function loadImage(path: string): Promise<HTMLImageElement> {
  return new Promise((resolve) => {
    const img = new Image();

    img.src = path;
    img.onload = () => {
      resolve(img);
    };
  });
}

function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
}
