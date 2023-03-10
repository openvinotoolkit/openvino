import { OpenvinoModule } from './ov-module.js';
import Session from './session.js';

import type { IShape, IModel, ISession } from './types.js';

export default class SessionBrowser extends Session implements ISession {
  constructor(ov: OpenvinoModule) {
    super(ov, 'browser');
  }

  async loadModel(xmlPath: Uint8Array | string, binPath: Uint8Array | string, shape: number[] | IShape, layout: string): Promise<IModel> {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string') {
      throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');
    }

    const xmlData = await getFileDataAsArray(xmlPath);  
    const binData = await getFileDataAsArray(binPath);

    if (!xmlData || !binData) throw new Error('Error on file uploading');

    return super.loadModel(xmlData, binData, shape, layout);
  }
}

async function getFileDataAsArray(path: string): Promise<Uint8Array | null> {
  const blob = await fetch(path).then(response => {
    if (!response.ok) {
      return null;
    }     
    return response.blob();
  });

  return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
}
