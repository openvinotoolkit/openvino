const {
  ImageData,
  loadImage,
  createCanvas,
} = require('@napi-rs/canvas');
const path = require('node:path');
const fs = require('node:fs/promises');
const { addon: ov } = require('openvino-node');

const codeENOENT = 'ENOENT';

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

  get grayscale() {
    const grayData = new Uint8ClampedArray(this.width * this.height);

    for (let i = 0; i < this.imageData.data.length; i += 4) {
      const [r, g, b] = this.imageData.data.slice(i, i + 3);
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;

      grayData[i / 4] = gray;
    }

    return grayData;
  }

  get canvasCtx() {
    const canvas = createCanvas(this.width, this.height);
    const ctx = canvas.getContext('2d');

    ctx.putImageData(this.imageData, 0, 0);

    return ctx;
  }

  get buffer() {
    return this.canvasCtx.canvas.toBuffer('image/jpeg');
  }

  drawRect(x, y, width, height, properties) {
    const ctx = this.canvasCtx;

    ctx.strokeStyle = properties.color || 'red';
    ctx.lineWidth = properties.width || 1;
    ctx.strokeRect(x, y, width, height);

    const imageData = ctx.getImageData(0, 0, this.width, this.height);

    return new OvImage(imageData);
  }

  drawText(text, x, y, properties) {
    const ctx = this.canvasCtx;

    ctx.font = properties.font || '30px Arial';
    ctx.fillStyle = properties.color || 'red';
    ctx.fillText(text, x, y);

    const imageData = ctx.getImageData(0, 0, this.width, this.height);

    return new OvImage(imageData);
  }

  drawCircle(x, y, radius, properties) {
    const ctx = this.canvasCtx;

    ctx.strokeStyle = properties.color || 'red';
    ctx.lineWidth = properties.width || 1;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.stroke();

    const imageData = ctx.getImageData(0, 0, this.width, this.height);

    return new OvImage(imageData);
  }

  drawLine(x1, y1, x2, y2, properties) {
    const ctx = this.canvasCtx;

    ctx.strokeStyle = properties.color || 'red';
    ctx.lineWidth = properties.width || 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const imageData = ctx.getImageData(0, 0, this.width, this.height);

    return new OvImage(imageData);
  }

  toTensor() {
    return new ov.Tensor(
      ov.element.u8,
      [1, this.height, this.width, 3],
      new Uint8ClampedArray(this.rgb),
    );
  }

  resize(newWidth, newHeight) {
    const ctx = this.canvasCtx;

    const canvas2 = createCanvas(newWidth, newHeight);
    const ctx2 = canvas2.getContext('2d');
    ctx2.drawImage(ctx.canvas, 0, 0, newWidth, newHeight);

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

  crop(x, y, width, height) {
    const canvas2 = createCanvas(width, height);
    const ctx2 = canvas2.getContext('2d');

    ctx2.drawImage(this.canvasCtx.canvas, x, y, width, height, 0, 0, width, height);

    const imageData = ctx2.getImageData(0, 0, width, height);

    return new OvImage(imageData);
  }

  async save(filepath) {
    const destination = path.dirname(filepath);

    try {
      await fs.access(destination);
    } catch(error) {
      if (error.code !== codeENOENT) throw error;

      await fs.mkdir(destination, { recursive: true });
    }

    return await fs.writeFile(filepath, this.buffer);
  }

  // Display the image using the node notebook display object
  display(display) {
    display.image(this.buffer);
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

  static overlay(img1, img2, alpha) {
    if (img1.width !== img2.width || img1.height !== img2.height)
      throw new Error('Images should have the same size');

    const img1Data = img1.imageData.data;
    const img2Data = img2.imageData.data;

    const overlayedData = img1Data.map((_, index) => {
      if (index % 4 === 3)
        return 255;

      return img1Data[index] * (1 - alpha) + img2Data[index] * alpha;
    });

    return OvImage.fromArray(overlayedData, img1.width, img1.height);
  }
}

module.exports = OvImage;
