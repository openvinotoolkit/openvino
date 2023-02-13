import { readFileSync } from 'node:fs';
import { createCanvas, loadImage } from 'canvas';

import ovWrapper from '../dist/ov_wrapper.mjs';
import openvinojs from '../dist/openvino_wasm.js';
import { getMaxElement } from '../dist/helpers.mjs';

import { default as imagenetClassesMap } from '../assets/imagenet_classes_map.mjs';

import Shape from '../../common/shape.mjs';
import Tensor from '../../common/tensor.mjs';

const MODEL_PATH = '../assets/models/';
const MODEL_NAME = 'v3-small_224_1.0_float';
const IMAGE_PATH = '../assets/images/coco224x224.jpg';

run();

async function testGetShape(openvinojs) {
  const ov = await openvinojs();

  const originalShapeObj = ov.getShape();
  // const dim = shapeObj.getDim();

  // console.log(dim);

  // const resultArray = [];
  // const heapResult = shapeObj.getData();

  // for (let i = 0; i < dim; i++) {
  //   resultArray.push(ov.HEAPF32[heapResult/Float32Array.BYTES_PER_ELEMENT + i]);
  // }

  const shape = Shape.parse(ov, originalShapeObj);

  console.log(`dim: ${shape.dim}`);
  console.log(`data: ${shape.data}`);
}

async function testProcessShape(openvinojs) {
  const ov = await openvinojs();

  const shape = new Shape(1, 224, 224, 3);
  const shape2 = new Shape(88, 99);

  const originalShape = shape.convert(ov);
  const originalShape2 = shape2.convert(ov);

  ov.processShape(originalShape.obj);
  ov.processShape(originalShape2.obj);

  originalShape.free();
  originalShape.free();
  ov.processShape(originalShape.obj);
  ov.processShape(originalShape2.obj);
}

async function getTensor(openvinojs) {
  const ov = await openvinojs();

  const originalTensor = ov.getTensor();

  const tensor = Tensor.parse(ov, originalTensor);
  const shape = tensor.shape;

  console.log(tensor.data);
  console.log(shape.data);

  return;
}

async function getShape2(openvinojs) {
  const ov = await openvinojs();

  const originalShape = ov.getShape2();
  const shape = Shape.parse(ov, originalShape);

  debugShape(ov, s);
}

function debugShape(ov, s) {
  console.log('= Debug Shape =');
  console.log(`== Dim: ${s.getDim()}`);

  const shape = Shape.parse(ov, s);
  console.log(`== dim: ${shape.dim}`);
  console.log(`== data: ${shape.data}`);

  console.log('= End Debug shape =');
}

async function run() {
  console.log('= Start');

  // await testGetShape(openvinojs);
  // await testProcessShape(openvinojs);

  await getTensor(openvinojs);
  // await getShape2(openvinojs);

  return;

  const ov = await ovWrapper.initialize(openvinojs);

  // console.log(`== OpenVINO v${ov.getVersionString()}`);
  // console.log(`== Description string: ${ov.getDescriptionString()}`);

  const xmlData = getFileDataAsArray(`${MODEL_PATH}${MODEL_NAME}.xml`);  
  const binData = getFileDataAsArray(`${MODEL_PATH}${MODEL_NAME}.bin`);  

  console.log('== before model loading');

  const model = await ov.loadModel(xmlData, binData, '[1, 224, 224, 3]', 'NHWC');

  console.log('== after model loading');

  const img = await getImgByPath(IMAGE_PATH);
  const imgData = await getArrayByImg(img);
  const imgTensor = new Uint8Array(imgData);

  const outputTensor = await model.run(imgTensor);

  console.log('== Output tensor:');
  console.log(outputTensor);

  const max = getMaxElement(outputTensor);
  console.log(`== Max index: ${max.index}, value: ${max.value}`);
  console.log(`== Result class: ${imagenetClassesMap[max.index]}`);

  console.log('= End');
}

function getImgByPath(path) {
  return loadImage(path);
}

async function getArrayByImg(image) {
  const { width, height } = image;

  const canvas = createCanvas(width, width);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
  const rgbaData = ctx.getImageData(0, 0, width, height).data;
  
  return rgbaData.filter((_, index) => (index + 1)%4);
}

function getFileDataAsArray(path) {
  const fileData = readFileSync(path);

  return new Uint8Array(fileData);
}
