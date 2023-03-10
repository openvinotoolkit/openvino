import { OpenvinoModule, OriginalModel } from './ov-module.js';
import Tensor from './tensor.js';

import type { ITensor, IModel } from './types.js';

export default class Model implements IModel {
  #ov: OpenvinoModule;
  #originalModel: OriginalModel;

  constructor(ov: OpenvinoModule, originalModel: OriginalModel) {
    this.#ov = ov;
    this.#originalModel = originalModel;
  }

  infer(tensor: ITensor): Promise<ITensor> {
    const wrapper = new Promise<ITensor>((resolve, reject) => {
      let outputTensor: ITensor | null;
      
      try {
        outputTensor = runInference(this.#ov, this.#originalModel, tensor);
      } catch(e) {
        return reject(e);
      }

      outputTensor ? resolve(outputTensor) : reject(null);
    });

    return wrapper;
  }
}

function runInference(ov: OpenvinoModule, model: OriginalModel, tensor: ITensor): ITensor | null {
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
