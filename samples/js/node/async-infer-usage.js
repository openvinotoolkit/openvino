const { addon: ov } = require('openvinojs-node');
const util = require('node:util');

const modelPath = './model.xml';
const inputData = []; // { [inputName: string]: Tensor }[]

const core = new ov.Core();
const model = core.readModel(modelPath);
const compiledModel = core.compileModel(model, 'CPU');

const inferRequest = compiledModel.createInferRequest();
const tensorData = new Float32Array([1.6041]);

const promisifiedAsyncInfer = util.promisify(ov.asyncInfer);
promisifiedAsyncInfer(inferRequest, [tensorData]).then((output) => {
    console.log("Output of the inference: ", output)
})
.catch((err) => {
    console.log("Error: ", err);
})


// inferRequest.asyncInfer(inputData[0], (err, result) => {
//   if (err) throw err; // Something went wrong

//   processResult(result); // result: { [outputTensorName: string]: Tensor }
// });

// const promises = inputData.map(i => {
//   const promisifiedAsyncInfer = util.promisify(inferRequest.asyncInfer); // : asyncInfer({ [inputName: string]: Tensor }) => Promise<{ [outputName: string]: Tensor }>

//   return promisifiedAsyncInfer(i); // : Promise<{ [outputName: string]: Tensor }>
// });
