import Tensor from "./tensor.mjs";

export default class Model {
  #ov = null;
  #originalModel = null;

  constructor(ov, originalModel, preprocess) {
    this.#ov = ov;
    this.#originalModel = originalModel;

    if (preprocess) this._preprocess = preprocess;
  }

  async infer(tensor) {
    const wrapper = new Promise((resolve, reject) => {
      let outputTensor = null;
      
      try {
        outputTensor = runInference(this.#ov, this.#originalModel, tensor);
      } catch(e) {
        return reject(e);
      }

      resolve(outputTensor);
    });

    return wrapper;
  }
}

function runInference(ov, model, tensor) {
  let originalTensor;
  let originalOutputTensor; 

  try {
    console.time('== Inference time');
    originalTensor = tensor.convert(ov);
    originalOutputTensor = model.infer(originalTensor.obj);
    console.timeEnd('== Inference time');
  } finally {
    if (originalTensor) originalTensor.free();
  }

  return originalOutputTensor ? Tensor.parse(ov, originalOutputTensor) : null;
}
