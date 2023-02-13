import ovWrapper from './ov_wrapper.mjs';
import { getMaxElement, getArrayByImgPath, getFileDataAsArray } from './helpers.mjs';

import { default as imagenetClassesMap } from '../assets/imagenet_classes_map.mjs';

export default async function run(Module, statusElement, modelPath, options, imgPath) {
  if (!options.shape) throw new Error('Model shape should be defined!');

  let xmlPath = '';
  let binPath = '';

  if (typeof modelPath === 'string') {
    xmlPath = modelPath;
    binPath = modelPath.replace(/.xml$/, '.bin');
  } else if (modelPath.xml && modelPath.bin) {
    xmlPath = modelPath.xml;
    binPath = modelPath.bin;
  }
 
  console.log('= Start');

  const ov = await ovWrapper.initialize(Module);

  console.log(`== OpenVINO v${ov.getVersionString()}`);
  console.log(`== Description string: ${ov.getDescriptionString()}`);

  statusElement.innerText = 'OpenVINO successfully initialized. Model loading...';
  const xmlData = await getFileDataAsArray(xmlPath);  
  const binData = await getFileDataAsArray(binPath);  

  const model = await ov.loadModel(xmlData, binData, options.shape, options.layout || 'NHWC');

  const imgData = await getArrayByImgPath(imgPath);
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

