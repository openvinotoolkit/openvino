import Shape from './shape.mjs';

export default class Session {
  _ov = null;
  _env = 'browser';

  constructor(ov, env) {
    this._ov = ov;

    if (env) this._env = env;
  }

  async loadModel(xmlData, binData, shapeData, layout) {
    const originalModel = await loadModel(this._ov, xmlData, binData, shapeData, layout);
    const ModelFactory = (await import(this._env == 'browser' ? './model-browser.mjs' : './model-nodejs.mjs')).default;

    return new ModelFactory(this._ov, originalModel, Session._preprocessInferParameters);
  }

  getVersionString() {
    return this._ov.getVersionString();
  }

  getDescriptionString() {
    return this._ov.getDescriptionString();
  }

  static async init(openvinojs, env = 'browser') {
    const SessionFactory = (await import(env == 'browser' ? './session-browser.mjs' : './session-nodejs.mjs')).default;

    return new SessionFactory(await openvinojs());
  }
}

function loadModel(ov, xmlData, binData, shapeData, layout) {
  const timestamp = Date.now();

  const xmlFilename = `m${timestamp}.xml`;
  const binFilename = `m${timestamp}.bin`;

  // Uploading and creating files on virtual WASM filesystem
  uploadFile(ov, xmlFilename, xmlData);
  uploadFile(ov, binFilename, binData);

  const shape = shapeData instanceof Shape ? shapeData : new Shape(...shapeData);
  const originalShape = shape.convert(ov);

  return new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);;
}

function uploadFile(ov, filename, data) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
