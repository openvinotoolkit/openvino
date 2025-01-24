const { app } = require('electron');
const { addon: ov } = require('openvino-node');
const { testModels, lengthFromShape } = require('../../utils.js');

const epsilon = 0.5; // To avoid very small numbers
const testModelFP32 = testModels.testModelFP32;

main();

async function main() {
  await app.whenReady();

  try {
    console.log('Creating OpenVINO Runtime Core');
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const core = new ov.Core();
    console.log('Created OpenVINO Runtime Core');

    const model = await core.readModel(testModelFP32.xml);
    console.log('Model read successfully:', model);
    const compiledModel = await core.compileModel(model, 'CPU');
    const inferRequest = compiledModel.createInferRequest();
    console.log('Infer request created:', inferRequest);

    const tensorData = Float32Array.from(
      { length: lengthFromShape(testModelFP32.inputShape) },
      () => Math.random() + epsilon,
    );
    const tensor = new ov.Tensor(ov.element.f32, testModelFP32.inputShape, tensorData);
    console.log('Tensor created:', tensor);

    const result = await inferRequest.inferAsync([tensor]);
    console.log('Infer request result:', result);
  } catch (error) {
    console.error('Error:', error);
    app.exit(1);
  }

  app.exit(0);
}
