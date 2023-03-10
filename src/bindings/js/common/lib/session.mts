import { OpenvinoModule, OriginalModel } from './ov-module.mjs';
import Shape from './shape.mjs';

import type { IShape, SessionEnvironment, ISession, IModel } from './types.mjs';

type ISessionFactory = new (ov: OpenvinoModule) => ISession
type IModelFactory = new (ov: OpenvinoModule, model: OriginalModel) => IModel

export default class Session implements ISession {
  _ov: OpenvinoModule;
  _env: SessionEnvironment = 'browser';

  constructor(ov: OpenvinoModule, env?: SessionEnvironment) {
    this._ov = ov;

    if (env) this._env = env;
  }

  async loadModel(xmlData: Uint8Array, binData: Uint8Array, shapeData: number[] | IShape, layout: string): Promise<IModel> {
    const originalModel = await loadModel(this._ov, xmlData, binData, shapeData, layout);
    const ModelFactory: IModelFactory = (await import(this._env == 'browser' ? './model-browser.mjs' : './model-nodejs.mjs')).default;

    return new ModelFactory(this._ov, originalModel);
  }

  getVersionString(): string {
    return this._ov.getVersionString();
  }

  getDescriptionString(): string {
    return this._ov.getDescriptionString();
  }

  static async init(openvinojs: () => Promise<OpenvinoModule>, env = 'browser'): Promise<ISession> {
    const SessionFactory: ISessionFactory = (await import(env == 'browser' ? './session-browser.mjs' : './session-nodejs.mjs')).default;

    return new SessionFactory(await openvinojs());
  }
};

function loadModel(ov: OpenvinoModule, xmlData: Uint8Array, binData: Uint8Array, shapeData: IShape | number[], layout: string): OriginalModel {
  const timestamp = Date.now();

  const xmlFilename = `m${timestamp}.xml`;
  const binFilename = `m${timestamp}.bin`;

  // Uploading and creating files on virtual WASM filesystem
  uploadFile(ov, xmlFilename, xmlData);
  uploadFile(ov, binFilename, binData);

  const shape = shapeData instanceof Shape ? shapeData : new Shape(...shapeData as number[]);
  const originalShape = shape.convert(ov);

  return new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);
}

function uploadFile(ov: OpenvinoModule, filename: string, data: Uint8Array) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
