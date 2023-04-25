import openvinoWASM from '../bin/openvino_wasm';
import { Tensor, Shape } from 'openvinojs-common';
import { 
  getFileDataAsArray, 
  uploadFile,
  convertShape,
  convertTensor,
  parseOriginalTensor,
} from './helpers';

import type { ITensor, IModel, IShape } from 'openvinojs-common';
import type { OpenvinoWASMModule, OriginalModel } from './types';

class WASMModel implements IModel {
  #ov: OpenvinoWASMModule;
  #originalModel: OriginalModel;

  constructor(ov: OpenvinoWASMModule, originalModel: OriginalModel) {
    this.#ov = ov;
    this.#originalModel = originalModel;
  }

  infer(tensorOrDataArray: ITensor | number[], shape: IShape): Promise<ITensor> {
    const tensor = tensorOrDataArray instanceof Tensor 
      ? tensorOrDataArray 
      : new Tensor('uint8', tensorOrDataArray as number[], shape);

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

export default async function loadModel(xmlPath: string, binPath: string, shapeData: Shape | number[], layout: string): Promise<IModel> {
  if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
    throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');

  const ov: OpenvinoWASMModule = await openvinoWASM();

  const xmlData = await getFileDataAsArray(xmlPath);
  const binData = await getFileDataAsArray(binPath);

  const timestamp = Date.now();

  const xmlFilename = `m${timestamp}.xml`;
  const binFilename = `m${timestamp}.bin`;

  // Uploading and creating files on virtual WASM filesystem
  uploadFile(ov, xmlFilename, xmlData);
  uploadFile(ov, binFilename, binData);

  const shape = shapeData instanceof Shape ? shapeData : new Shape(...shapeData as number[]);
  const originalShape = convertShape(ov, shape);

  const originalModel = new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);

  return new WASMModel(ov, originalModel);
}

function runInference(ov: OpenvinoWASMModule, model: OriginalModel, tensor: ITensor): ITensor | null {
  let originalTensor;
  let originalOutputTensor; 

  try {
    console.time('== Inference time');
    originalTensor = convertTensor(ov, tensor);
    originalOutputTensor = model.infer(originalTensor.obj);
    console.timeEnd('== Inference time');
  } finally {
    if (originalTensor) originalTensor.free();
  }

  return originalOutputTensor ? parseOriginalTensor(ov, originalOutputTensor) : null;
}
