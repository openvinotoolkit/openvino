const { downloadFile } = require('./utils.js');

if (require.main === module) {
  main();
}

async function main() {
  const baseArtifactsDir = './test_models';
  const modelName = 'v3-small_224_1.0_float';
  const modelXMLName = `${modelName}.xml`;
  const modelBINName = `${modelName}.bin`;
  const baseURL = 'https://storage.openvinotoolkit.org/repositories/'
    + 'openvino_notebooks/models/mobelinet-v3-tf/FP32/';

  await downloadFile(baseURL + modelXMLName, modelXMLName, baseArtifactsDir);
  await downloadFile(baseURL + modelBINName, modelBINName, baseArtifactsDir);

}
