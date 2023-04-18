import openvinoWASM from '../bin/openvino_wasm';
import loadModel from './wasm-model';
import { Tensor, Shape } from 'openvinojs-common';

import type { OpenvinoModule } from './types';

export async function init() { 
  const ov: OpenvinoModule = await openvinoWASM();

  function getVersionString(): string {
    return ov.getVersionString();
  }
  
  function getDescriptionString(): string {
    return ov.getDescriptionString();
  }

  return {
    loadModel, 
    Tensor,
    Shape,
    getVersionString, 
    getDescriptionString,
  };
};
