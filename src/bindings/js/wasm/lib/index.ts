import openvinoWASM from '../bin/openvino_wasm';
import loadModel from './wasm-model';

import createModule from 'openvinojs-common';

export default createModule('wasm', loadModel, getVersionString, getDescriptionString);

async function getVersionString(): Promise<string> {
  const ov = await openvinoWASM();
  
  return ov.getVersionString();
}

async function getDescriptionString(): Promise<string> {
  const ov = await openvinoWASM();
  
  return ov.getDescriptionString();
}
