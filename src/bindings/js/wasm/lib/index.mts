import ov from './openvino_wasm.mjs';
import { Tensor, Model, Shape } from 'openvinojs-common';
import { getFileDataAsArray, uploadFile } from './helpers.mjs';

class OriginalModel {
  constructor()
}

async function loadModel(xmlPath: string, binPath: string, shapeData: Shape | number[], layout: string): Model {
  const xmlData = await getFileDataAsArray(xmlPath);
  const binData = await getFileDataAsArray(binPath);

  const timestamp = Date.now();

  const xmlFilename = `m${timestamp}.xml`;
  const binFilename = `m${timestamp}.bin`;

  // Uploading and creating files on virtual WASM filesystem
  uploadFile(ov, xmlFilename, xmlData);
  uploadFile(ov, binFilename, binData);

  const originalModel = new OriginalModel();

  return new Model(originalModel);
}

export { loadModel, Tensor };
