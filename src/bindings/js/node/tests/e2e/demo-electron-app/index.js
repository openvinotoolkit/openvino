const { app } = require('electron');
const { addon: ov } = require('openvino-node');

const epsilon = 0.5; // To avoid very small numbers
const pathToModel = '../tests/unit/test_models/test_model_fp32.xml';

main();

async function main() {
  await app.whenReady();

  try {
    console.log('Creating OpenVINO Runtime Core');
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const core = new ov.Core();
    console.log('Created OpenVINO Runtime Core');

    const model = await core.readModel(pathToModel);
    console.log('Model read successfully:', model);
    const compiledModel = await core.compileModel(model, 'CPU');
    const inferRequest = compiledModel.createInferRequest();
    console.log('Infer request created:', inferRequest);

    const tensorData = Float32Array.from(
      { length: 3072 },
      () => Math.random() + epsilon,
    );
    const tensor = new ov.Tensor(ov.element.f32, [1, 3, 32, 32], tensorData);
    console.log('Tensor created:', tensor);

    const result = await inferRequest.inferAsync([tensor]);
    console.log('Infer request result:', result);
  } catch (error) {
    console.error('Error:', error);
    app.exit(1);
  }

  app.exit(0);
}
