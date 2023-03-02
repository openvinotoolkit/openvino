import { readFileSync } from 'node:fs';

import Session from './session.mjs';

export default class SessionNodejs extends Session {
  constructor(ov) {
    super(ov, 'nodejs');
  }

  loadModel(xmlPath, binPath, shape, layout) {
    const xmlData = getFileDataAsArray(xmlPath);  
    const binData = getFileDataAsArray(binPath);

    return super.loadModel(xmlData, binData, shape, layout);
  }
}

function getFileDataAsArray(path) {
  const fileData = readFileSync(path);

  return new Uint8Array(fileData);
}
