const { app } = require("electron");

const epsilon = 0.5; // To avoid very small numbers
const pathToModel = "../tests/unit/test_models/test_model_fp32.xml";
const scenario = process.env.OV_ELECTRON_SCENARIO ?? "zero-copy-async";
const supportedScenarios = new Set([
  "empty",
  "addon",
  "compile",
  "owned-async",
  "zero-copy",
  "zero-copy-sync",
  "zero-copy-async",
]);

main();

async function main() {
  await app.whenReady();

  try {
    if (!supportedScenarios.has(scenario)) {
      throw new Error(`Unsupported E2E scenario: ${scenario}`);
    }

    console.error(`[OV_E2E] scenario=${scenario} stage=ready`);
    if (scenario === "empty") {
      finishScenario();

      return;
    }

    const { addon: ov } = require("openvino-node");
    console.error(`[OV_E2E] scenario=${scenario} stage=addon-loaded`);
    if (scenario === "addon") {
      finishScenario();

      return;
    }

    if (scenario === "zero-copy") {
      createTensor(ov, true);
      console.log("Tensor created");
      finishScenario();

      return;
    }

    console.log("Creating OpenVINO Runtime Core");
    const core = new ov.Core();
    console.log("Created OpenVINO Runtime Core");

    const model = await core.readModel(pathToModel);
    console.log("Model read successfully:", model);
    const compiledModel = await core.compileModel(model, "CPU");
    console.error(`[OV_E2E] scenario=${scenario} stage=model-compiled`);
    if (scenario === "compile") {
      finishScenario();

      return;
    }

    const inferRequest = compiledModel.createInferRequest();
    console.log("Infer request created:", inferRequest);

    const tensor = createTensor(ov, scenario !== "owned-async");
    console.log("Tensor created:", tensor);

    const result =
      scenario === "zero-copy-sync"
        ? inferRequest.infer([tensor])
        : await inferRequest.inferAsync([tensor]);
    console.log("Infer request result:", result);
    finishScenario();
  } catch (error) {
    console.error("Error:", error);
    app.exit(1);

    return;
  }
}

function createTensor(ov, zeroCopy) {
  const shape = [1, 3, 32, 32];
  if (!zeroCopy) {
    return new ov.Tensor(ov.element.f32, shape);
  }

  const tensorData = Float32Array.from({ length: 3072 }, () => Math.random() + epsilon);
  return new ov.Tensor(ov.element.f32, shape, tensorData);
}

function finishScenario() {
  console.error(`[OV_E2E] scenario=${scenario} stage=app-exit`);
  console.log(`Scenario completed: ${scenario}`);
  app.exit(0);
}
