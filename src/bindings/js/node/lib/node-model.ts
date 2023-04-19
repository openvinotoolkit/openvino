import { Tensor, Shape } from 'openvinojs-common';
import type { ITensor, IModel, IShape } from 'openvinojs-common';

export default async function loadModel(xmlPath: string, binPath: string, shapeData: Shape | number[], layout: string): Promise<IModel> {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
      throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');
  
    // const ov: OpenvinoModule = await openvinoWASM();
  
    // const xmlData = await getFileDataAsArray(xmlPath);
    // const binData = await getFileDataAsArray(binPath);
  
    // const timestamp = Date.now();
  
    // const xmlFilename = `m${timestamp}.xml`;
    // const binFilename = `m${timestamp}.bin`;
  
    // // Uploading and creating files on virtual WASM filesystem
    // uploadFile(ov, xmlFilename, xmlData);
    // uploadFile(ov, binFilename, binData);
  
    // const shape = shapeData instanceof Shape ? shapeData : new Shape(...shapeData as number[]);
    // const originalShape = convertShape(ov, shape);
  
    // const originalModel = new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);
  
    // return new WASMModel(ov, originalModel);
  }
  