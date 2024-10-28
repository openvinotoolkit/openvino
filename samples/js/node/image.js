const {
  ImageData,
  loadImage,
  createCanvas,
} = require('canvas');
const fs = require('node:fs/promises');
const { addon: ov } = require('openvino-node');

class OvImage {
  constructor(imageData) {
    this.imageData = imageData;
    this.channels = imageData.data.length / (this.width*this.height);
  }

  get width() {
    return this.imageData.width;
  }

  get height() {
    return this.imageData.height;
  }

  get rgb() {
    return this.imageData.data.filter((_, index) => index % 4 !== 3);
  }

  get rgba() {
    return this.imageData.data;
  }

  toTensor() {
    return new ov.Tensor(
      ov.element.u8,
      [1, this.height, this.width, 3],
      new Uint8ClampedArray(this.rgb),
    );
  }

  resize(newWidth, newHeight) {
    const canvas = createCanvas(this.width, this.height);
    const ctx = canvas.getContext('2d');

    ctx.putImageData(this.imageData, 0, 0);

    const canvas2 = createCanvas(newWidth, newHeight);
    const ctx2 = canvas2.getContext('2d');
    ctx2.drawImage(canvas, 0, 0, newWidth, newHeight);

    const imageData = ctx2.getImageData(0, 0, newWidth, newHeight);

    return new OvImage(imageData);
  }

  invert() {
    const invertedData = this.rgba.map((value, index) => {
      if (index % 4 === 3)
        return 255;

      return 255 - value;
    });

    return OvImage.fromArray(invertedData, this.width, this.height);
  }

  async save(path) {
    const canvas = createCanvas(this.width, this.height);
    const ctx = canvas.getContext('2d');

    ctx.putImageData(this.imageData, 0, 0);

    const buffer = canvas.toBuffer('image/jpeg');

    return await fs.writeFile(path, buffer);
  }

  static async load(path) {
    const image = await loadImage(path);
    const { width, height } = image;

    const canvas = await createCanvas(width, height);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0);

    return new OvImage(ctx.getImageData(0, 0, width, height));
  }

  static fromArray(arr, width, height) {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');

    const imageData = new ImageData(
      new Uint8ClampedArray(arr),
      width,
      height,
    );

    ctx.putImageData(imageData, 0, 0);

    return new OvImage(ctx.getImageData(0, 0, width, height));
  }

  static merge(img1, img2) {
    if (img1.width !== img2.width || img1.height !== img2.height)
      throw new Error('Images should have the same size');

    const canvas = createCanvas(img1.width, img1.height);
    const ctx = canvas.getContext('2d');

    const img1Data = img1.imageData.data;
    const img2Data = img2.imageData.data;

    const mergedData = img1Data.map((_, index) => {
      if (index % 4 === 3)
        return 255;

      return (img1Data[index] + img2Data[index]);
    });

    const imageData = new ImageData(
      new Uint8ClampedArray(mergedData),
      img1.width,
      img1.height,
    );

    ctx.putImageData(imageData, 0, 0);

    return new OvImage(ctx.getImageData(0, 0, img1.width, img1.height));
  }

  static mask(img1, img2) {
    if (img1.width !== img2.width || img1.height !== img2.height)
      throw new Error('Images should have the same size');

    const canvas = createCanvas(img1.width, img1.height);
    const ctx = canvas.getContext('2d');

    const img1Data = img1.imageData.data;
    const img2Data = img2.imageData.data;

    const subtractedData = img1Data.map((_, index) => {
      if (index % 4 === 3)
        return 255;

      return img1Data[index] * (img2Data[index] / 255);
    });

    const imageData = new ImageData(
      new Uint8ClampedArray(subtractedData),
      img1.width,
      img1.height,
    );

    ctx.putImageData(imageData, 0, 0);

    return new OvImage(ctx.getImageData(0, 0, img1.width, img1.height));
  }
}

module.exports = OvImage;
