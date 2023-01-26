export function getMaxElement(arr) {
  if (!arr.length) return { value: -Infinity, index: -1 };

  let max = arr[0];
  let maxIndex = 0;

  for (let i = 1; i < arr.length; ++i) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }

  return { value: max, index: maxIndex };
}

export async function getFileDataAsArray(path) {
  const blob = await fetch(path).then(response => {
    if (!response.ok) {
      return null;
    }     
    return response.blob();
  });

  return new Uint8Array(await blob.arrayBuffer());
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

export function createCanvas(width, height) {
  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
}

export async function getArrayByImgPath(path) {
  const image = await loadImage(path);
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  return rgbaData.filter((_, index) => (index + 1)%4);
}
