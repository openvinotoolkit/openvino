const { app } = require("electron");

const epsilon = 0.5; // To avoid very small numbers
const modelName = process.env.OV_E2E_MODEL ?? "test-model-fp32";
const modelPaths = new Map([
  ["test-model-fp32", "../tests/unit/test_models/test_model_fp32.xml"],
  ["add-model", "../tests/unit/test_models/add_model.xml"],
  ["relu-model", "../tests/unit/test_models/relu_model.xml"],
]);
const scenario = process.env.OV_ELECTRON_SCENARIO ?? "zero-copy-async";
const supportedScenarios = new Set([
  "empty",
  "addon",
  "core",
  "cpu-plugin",
  "read-sync",
  "read-async",
  "compile-sync",
  "compile-async",
  "owned-async",
  "zero-copy",
  "zero-copy-sync",
  "zero-copy-async",
]);
const syncReadScenarios = new Set(["read-sync", "compile-sync", "compile-async"]);

main();

async function main() {
  await app.whenReady();

  try {
    if (!supportedScenarios.has(scenario)) {
      throw new Error(`Unsupported E2E scenario: ${scenario}`);
    }
    if (!modelPaths.has(modelName)) {
      throw new Error(`Unsupported E2E model: ${modelName}`);
    }

    console.error(`[OV_E2E] scenario=${scenario} model=${modelName} stage=ready`);
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
    if (scenario === "core") {
      finishScenario();

      return;
    }
    if (scenario === "cpu-plugin") {
      const versions = core.getVersions("CPU");
      console.error(`[OV_E2E] scenario=${scenario} model=${modelName} stage=plugin-loaded`);
      console.log("CPU plugin loaded successfully:", versions);
      finishScenario();

      return;
    }

    const syncRead = syncReadScenarios.has(scenario);
    const pathToModel = modelPaths.get(modelName);
    const model = syncRead ? core.readModelSync(pathToModel) : await core.readModel(pathToModel);
    console.error(
      `[OV_E2E] scenario=${scenario} model=${modelName} stage=model-read mode=${syncRead ? "sync" : "async"}`,
    );
    console.log("Model read successfully:", model);
    if (scenario === "read-sync" || scenario === "read-async") {
      finishScenario();

      return;
    }

    const syncCompile = scenario === "compile-sync";
    const compiledModel = syncCompile
      ? core.compileModelSync(model, "CPU")
      : await core.compileModel(model, "CPU");
    console.error(
      `[OV_E2E] scenario=${scenario} model=${modelName} stage=model-compiled mode=${syncCompile ? "sync" : "async"}`,
    );
    if (scenario === "compile-sync" || scenario === "compile-async") {
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
  console.error(`[OV_E2E] scenario=${scenario} model=${modelName} stage=app-exit`);
  console.log(`Scenario completed: ${scenario} model=${modelName}`);
  app.exit(0);
}
