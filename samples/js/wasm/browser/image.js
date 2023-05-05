const selectBtn = document.getElementById('select-btn');
const containerElement = document.getElementById('container');

const statusElement = document.getElementById('status');
const panelElement = document.getElementById('panel');

const MODEL_PATH = '../assets/models/';
const MODEL_NAME = 'v3-small_224_1.0_float';

const WIDTH = 224;
const HEIGHT = 224;

const WAITING_INPUT_STATUS_TXT =
  'OpenVINO initialized. Model loaded. Select image to make an inference';

run();

async function run() {
  const { Shape, loadModel } = openvinojs;

  statusElement.innerText =
    'OpenVINO successfully initialized. Model loading...';

  const shape = new Shape(1, WIDTH, HEIGHT, 3);
  const model = await loadModel(
    {
      path: MODEL_PATH,
      modelName: MODEL_NAME,
    },
    shape,
    'NHWC',
  );

  statusElement.innerText = WAITING_INPUT_STATUS_TXT;
  panelElement.classList.remove('hide');

  selectBtn.addEventListener('change', fileSelectorHandler);

  async function fileSelectorHandler(e) {
    containerElement.innerHTML = '';

    const selectedFile = e.srcElement.files[0];

    const fileData = await imgWasSelected(selectedFile);
    const img = new Image();
    const canvas = createCanvas(WIDTH, HEIGHT);
    const ctx = canvas.getContext('2d');

    img.src = fileData;
    img.addEventListener('load', async () => {
      console.log('== image loaded');

      containerElement.appendChild(img);
      containerElement.appendChild(canvas);

      ctx.drawImage(img, 0, 0, WIDTH, HEIGHT);

      statusElement.innerText = 'Inference is in the progress, please wait...';
      const imgTensor = getImgTensor(ctx);
      const startTime = performance.now();
      const outputTensor = await model.infer(imgTensor, shape);
      const endTime = performance.now();
      statusElement.innerText = WAITING_INPUT_STATUS_TXT;

      console.log('== Output tensor:');
      console.log(outputTensor);

      const max = getMaxElement(outputTensor.data);
      console.log(`== Max index: ${max.index}, value: ${max.value}`);
      const imagenetClassesMap =
        await fetch('./assets/imagenet_classes_map.json').then(
          response => response.json(),
        );
      const humanReadableClass = imagenetClassesMap[max.index];
      console.log(`== Result class: ${humanReadableClass}`);

      const infoElement = getInfoElement({
        time: endTime - startTime,
        className: humanReadableClass,
        index: max.index,
        value: max.value,
      });
      containerElement.append(infoElement);

      console.log('= End');
    });
  }

  function getImgTensor(canvasCtx) {
    const rgbaData = canvasCtx.getImageData(0, 0, WIDTH, HEIGHT).data;

    return rgbaData.filter((_, index) => (index + 1)%4);
  }
}

function imgWasSelected(file) {
  const fr = new FileReader();

  fr.readAsDataURL(file);

  return new Promise((resolve, reject) => {
    fr.addEventListener('load', () => {
      resolve(fr.result);
    });

    fr.addEventListener('error', () => {
      reject(new Error('Error on loading file'));
    });
  });
}

function getInfoElement({ time, className, index, value }) {
  const e = document.createElement('pre');

  e.innerText = `\
    Inference time: ${Number(time).toFixed(3)}ms\n\
    Class: ${className}\n\
    Index: ${index}\n\
    Value: ${value}`;

  return e;
}

function getMaxElement(arr) {
  if (!arr.length) return { value: -Infinity, index: -1 };

  let max = arr[0];
  let maxIndex = 0;

  for (let i = 1; i < arr.length; ++i)
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }

  return { value: max, index: maxIndex };
}

function createCanvas(width, height) {
  const canvasElement = document.createElement('canvas');

  canvasElement.width = width;
  canvasElement.height = height;

  return canvasElement;
}
