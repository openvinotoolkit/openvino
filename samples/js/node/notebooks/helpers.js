const { display } = require('node-kernel');
const { createCanvas, createImageData, loadImage, Image, ImageData } = require('canvas');

exports.arrayToImageData = arrayToImageData;
exports.displayImage = displayImage;
exports.getImageData = getImageData;
exports.displayArrayAsImage = displayArrayAsImage;
exports.transform = transform;

function arrayToImageData(array, width, height) {
  return createImageData(new Uint8ClampedArray(array), width, height);
}

function displayImage(imageOrImageData) {
  const canvas = createCanvas(imageOrImageData.width, imageOrImageData.height);
  const ctx = canvas.getContext('2d');

  if (imageOrImageData instanceof Image)
    ctx.drawImage(imageOrImageData, 0, 0);
  else if (imageOrImageData instanceof ImageData)
    ctx.putImageData(imageOrImageData, 0, 0);
  else
    throw Error(`Passed parameters has type '${typeof imageOrImageData}'. It is't supported.`);

  const buffer = canvas.toBuffer('image/jpeg');

  display.image(buffer);
}

function displayArrayAsImage(arr, width, height) {
  const alpha = 255;
  const componentsPerPixel = arr.length/(width*height);

  try {
    switch(componentsPerPixel) {
      case 1:
        arr = arr.reduce((acc, val) => {
          acc.push(val, val, val, alpha);

          return acc;
        }, []);
        break;
      case 3:
        arr = arr.reduce((acc, val, index) => {
          if (index && index%3 === 0) acc.push(alpha);

          acc.push(val);

          return acc;
        }, []);
        break;
    }
  } catch(e) {
    console.log(e);
  }

  const imageData = arrayToImageData(arr, width, height);

  displayImage(imageData);
}

async function getImageData(path) {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = await createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);

  return ctx.getImageData(0, 0, width, height);
}

function transform(arr, { width, height }, order) {
  const img = new cv2.Mat(height, width, cv2.CV_8UC3);

  img.data.set(arr, 0, arr.length);

  const channels = new cv2.MatVector();
  cv2.split(img, channels);

  const val = order.map(num => [...channels.get(num).data]);

  return [].concat(...val);
}
