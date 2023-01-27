import ovWrapper from './dist/ov_wrapper.mjs';
import { getMaxElement, getArrayByImgPath, getFileDataAsArray } from './dist/helpers.mjs';

import { default as imagenetClassesMap } from '../assets/imagenet_classes_map.mjs';

const MODEL_PATH = '../assets/models/';
const MODEL_NAME = 'v3-small_224_1.0_float';
const IMAGE_PATH = '../assets/images/coco.jpg';

const statusElement = document.getElementById('status');

run();

async function run() {
  console.log('= Start');

  const ov = await ovWrapper.initialize(Module);

  console.log(`== OpenVINO v${ov.getVersionString()}`);
  console.log(`== Description string: ${ov.getDescriptionString()}`);

  statusElement.innerText = 'OpenVINO successfully initialized. Model loading...';
  const xmlData = await getFileDataAsArray(`${MODEL_PATH}${MODEL_NAME}.xml`);  
  const binData = await getFileDataAsArray(`${MODEL_PATH}${MODEL_NAME}.bin`);  

  const model = await ov.loadModel(xmlData, binData, '[1, 224, 224, 3]', 'NHWC');

  const imgData = await getArrayByImgPath(IMAGE_PATH);
  const imgTensor = new Uint8Array(imgData);

  statusElement.innerText = 'Inference is in the progress, please wait...';
  const outputTensor = await model.run(imgTensor);

  statusElement.innerText = 'Open browser\'s console to see result';
  console.log('== Output tensor:');
  console.log(outputTensor);

  const max = getMaxElement(outputTensor);
  console.log(`== Max index: ${max.index}, value: ${max.value}`);
  console.log(`== Result class: ${imagenetClassesMap[max.index]}`);

  console.log('= End');
}

