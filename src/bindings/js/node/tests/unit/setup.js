const { testModels, downloadTestModel } = require('./utils.js');

async function main() {
  await downloadTestModel(testModels.testModelFP32);
}

if (require.main === module) {
  main();
}
