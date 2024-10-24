const {
  Image,
  ImageData,
  loadImage,
  createCanvas,
} = require('canvas');
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
}

module.exports = OvImage;
