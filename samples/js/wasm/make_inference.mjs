import { Session, Tensor } from 'openvinojs-wasm';
import { default as imagenetClassesMap } from './assets/imagenet_classes_map.mjs';

export default async function({ modelPath, imgPath, shape, layout }, events = {}) {
  events.onLibInitializing = events.onLibInitializing || (() => {});
  events.onModelLoaging = events.onModelLoaging || (() => {});
  events.onInferenceRunning = events.onInferenceRunning || (() => {});
  events.onFinish = events.onFinish || (() => {});

  console.log('= Start');

  events.onLibInitializing();
  const session = await Session.init();

  console.log(`== OpenVINO v${session.getVersionString()}`);
  console.log(`== Description string: ${session.getDescriptionString()}`);

  events.onModelLoaging(session);
  const model = await session.loadModel(modelPath.xml, modelPath.bin, shape, layout);

  events.onInferenceRunning(model);
  const inputTensor = getTensorByImgPath(imgPath);
  const outputTensor = await model.infer(inputTensor, shape);

  events.onFinish(outputTensor);
  const max = getMaxElement(outputTensor.data);
  console.log(`== Max index: ${max.index}, value: ${max.value}`);
  console.log(`== Result class: ${imagenetClassesMap[max.index]}`);

  console.log('= End');
}

function getMaxElement(arr) {
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

function getTensorByImgPath(path) {

}

function isNodeEnv() {
  return import.meta.url.startsWith('file:');
}
