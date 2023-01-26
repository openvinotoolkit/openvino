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

    const session = new ov.Session(xmlFilename, binFilename, shape, layout);

    return { run: runInference(ov, session) };
  };
}

function runInference(ov, session) {
  return inputData => {
    let heapSpace = null;
    let heapResult = null;
    try {
      heapSpace = ov._malloc(inputData.length*inputData.BYTES_PER_ELEMENT);
      ov.HEAPU8.set(inputData, heapSpace); 

      console.time('== Inference time');
      heapResult = session.run(heapSpace, inputData.length);
      console.timeEnd('== Inference time');
    } finally {
      ov._free(heapSpace);
    }

    const outputTensorSize = session.outputTensorSize;
    const outputTensorData = [];
    for (let v = 0; v < outputTensorSize; v++) {
      outputTensorData.push(ov.HEAPF32[heapResult/Float32Array.BYTES_PER_ELEMENT + v]);
    }

    return outputTensorData;
  };
}

function uploadFile(ov, filename, data) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
