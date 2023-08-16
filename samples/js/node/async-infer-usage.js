const { addon: ov } = require('openvinojs-node');
const util = require('node:util');

const modelPath = '../assets/models/v3-small_224_1.0_float.xml';
const inputData = [
  Float32Array.from({length: 150528}, () => Math.random() ),
  Float32Array.from({length: 150528}, () => Math.random() )];
  // e.g. [tensorData1: TypedArray, tensorData2: TypedArray, ...]

const core = new ov.Core();
const model = core.readModel(modelPath);
const compiledModel = core.compileModel(model, 'CPU');

const inferRequest = compiledModel.createInferRequest();

const promises = inputData.map(i => {
  // : asyncInfer({ inferRequest: InferRequest, [inputName: string]: Tensor })
  // => Promise<{ [outputName: string]: Tensor }>
  const promisifiedAsyncInfer = util.promisify(ov.asyncInfer);
  
  // : Promise<{ [outputName: string]: Tensor }>
  return promisifiedAsyncInfer(inferRequest, [i]);
  
});

Promise.all(promises).then(outputs => {
  for (const i in outputs)
    console.log(outputs[i]);
}).catch((err) => {
  console.log('Error: ', err);
});
