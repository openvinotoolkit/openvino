import { readFileSync } from 'node:fs';
import { OpenvinoModule } from './ov-module.js';

import Session from './session.js';

import type { IShape } from './types.js';

export default class SessionNodejs extends Session {
  constructor(ov: OpenvinoModule) {
    super(ov, 'nodejs');
  }

  loadModel(xmlPath: Uint8Array | string, binPath: Uint8Array | string, shape: number[] | IShape, layout: string) {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string') {
      throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');
    }

    const xmlData = getFileDataAsArray(xmlPath);  
    const binData = getFileDataAsArray(binPath);

    return super.loadModel(xmlData, binData, shape, layout);
  }
}

function getFileDataAsArray(path: string): Uint8Array {
  const fileData = readFileSync(path);

  return new Uint8Array(fileData);
}
