import Tensor from "./tensor.mjs";

export default { initialize };

async function initialize(openvinojs) {
  const ov = await openvinojs();

  return { 
    loadModel: loadModel(ov),
    getVersionString: () => ov.getVersionString(),
    getDescriptionString: () => ov.getDescriptionString(),
  };
}

function loadModel(ov) {
  return async (xmlData, binData, shape, layout) => {
    const timestamp = Date.now();

    const xmlFilename = `m${timestamp}.xml`;
    const binFilename = `m${timestamp}.bin`;

    // Uploading and creating files on virtual WASM filesystem
    uploadFile(ov, xmlFilename, xmlData);
    uploadFile(ov, binFilename, binData);

    const originalShape = shape.convert(ov);
    const session = new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);

    // Do not freeze UI wrapper
    return { 
      infer: (tensor) => {
        const infer = runInference(ov, session);
        
        return new Promise(resolve => {
          setTimeout(() => resolve(infer(tensor)), 0)
        });
      }
    };
  };
}

function runInference(ov, session) {
  return tensor => {
    let originalTensor;
    let originalOutputTensor; 

    try {
      console.time('== Inference time');
      originalTensor = tensor.convert(ov);
      originalOutputTensor = session.infer(originalTensor.obj);
      console.timeEnd('== Inference time');
    } finally {
      if (originalTensor) originalTensor.free();
    }

    return originalOutputTensor ? Tensor.parse(ov, originalOutputTensor) : null;
  };
}

function uploadFile(ov, filename, data) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
