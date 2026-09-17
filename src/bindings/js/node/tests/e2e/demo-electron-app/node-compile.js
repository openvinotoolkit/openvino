const modelName = process.env.OV_E2E_MODEL ?? "test-model-fp32";
const compileMode = process.env.OV_NODE_COMPILE_MODE ?? "async";
const modelPaths = new Map([
  ["test-model-fp32", "../tests/unit/test_models/test_model_fp32.xml"],
  ["add-model", "../tests/unit/test_models/add_model.xml"],
  ["relu-model", "../tests/unit/test_models/relu_model.xml"],
]);

main().catch((error) => {
  console.error("Error:", error);
  process.exitCode = 1;
});

async function main() {
  if (!modelPaths.has(modelName)) {
    throw new Error(`Unsupported E2E model: ${modelName}`);
  }
  if (compileMode !== "sync" && compileMode !== "async") {
    throw new Error(`Unsupported Node compile mode: ${compileMode}`);
  }

  console.error(`[OV_NODE_E2E] model=${modelName} mode=${compileMode} stage=ready`);
  const { addon: ov } = require("openvino-node");
  const core = new ov.Core();
  const pathToModel = modelPaths.get(modelName);
  const model =
    compileMode === "sync" ? core.readModelSync(pathToModel) : await core.readModel(pathToModel);
  const compiledModel =
    compileMode === "sync"
      ? core.compileModelSync(model, "CPU")
      : await core.compileModel(model, "CPU");

  console.error(`[OV_NODE_E2E] model=${modelName} mode=${compileMode} stage=model-compiled`);
  console.log("Compiled model:", compiledModel);
  console.log(`Node compile completed: model=${modelName} mode=${compileMode}`);
}
