import { printShape, getArrayByImgPath } from '../common/index.mjs';

const modelPath =
  '../assets/models/face-detection-0200/face-detection-0200.xml';
const imgPath = '../assets/images/peopleAndCake256x256.jpg';
const shape = [1, 3, 256, 256];
const layout = 'NCHW';

export default async function(openvinojs) {
  const model = await openvinojs.loadModel(modelPath, shape, layout);
  const inputTensor = await getArrayByImgPath(imgPath);
  const outputTensor = await model.infer(inputTensor, shape);

  printShape(outputTensor.shape);
}
