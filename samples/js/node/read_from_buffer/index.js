const { addon } = require('openvinojs-node');

const core = new addon.Core();

// this.#inferenceSession.loadModel(pathOrBuffer.buffer, pathOrBuffer.byteOffset, pathOrBuffer.byteLength, options);

const buffer = new ArrayBuffer(8);

core.readModelFromBuffer(buffer, buffer.byteOffset, buffer.byteLength);
