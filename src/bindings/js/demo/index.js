const fs = require('fs');
const openvinojs = require('../../../../bin/ia32/Release/openvino_wasm.js');

const FILENAME = 'v3-small_224_1.0_float';

openvinojs().then(async ov => {
  console.log('== start');

  console.log(ov.getVersionString());
  console.log(ov.getDescriptionString());

  // Uploading and creating files on virtual WASM fs
  await uploadFile(`../tmp/models/${FILENAME}.bin`, ov);
  await uploadFile(`../tmp/models/${FILENAME}.xml`, ov);

  const status = ov.readModel(`${FILENAME}.xml`, `${FILENAME}.bin`);
  console.log(status);
  
  console.log('== end');
});

async function uploadFile(path, ov) {
  const filename = path.split('/').pop();
  const fileData = fs.readFileSync(path);

  const data = new Uint8Array(fileData);

  const stream = ov.FS.open(filename, 'w+');
  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
