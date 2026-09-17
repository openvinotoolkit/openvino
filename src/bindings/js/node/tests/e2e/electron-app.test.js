/* global describe, it, before, after */
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");
const util = require("node:util");
const assert = require("node:assert");
const { exec, spawn } = require("node:child_process");
const execPromise = util.promisify(exec);
const { testModels, downloadTestModel } = require("../utils.js");

const appDirectory = "demo-electron-app-project";
const fullScenario = "zero-copy-async";
const diagnosticScenarios = [
  "empty",
  "addon",
  "compile",
  "owned-async",
  "zero-copy",
  "zero-copy-sync",
  fullScenario,
];
const diagnosticsEnabled = process.env.OPENVINO_E2E_DIAGNOSTICS === "1";
const scenarios = diagnosticsEnabled ? diagnosticScenarios : [fullScenario];
const inferRequestScenarios = new Set(["owned-async", fullScenario]);
const tensorScenarios = new Set(["zero-copy", "zero-copy-sync", fullScenario]);
const tensorCleanupScenarios = new Set(["zero-copy-sync", fullScenario]);

describe("E2E testing for OpenVINO as an Electron dependency.", function () {
  this.timeout(diagnosticsEnabled ? 120000 : 50000);

  before(async () => {
    await downloadTestModel(testModels.testModelFP32);
    await execPromise(`cp -r ./tests/e2e/demo-electron-app/ ${appDirectory}`);
  });

  it("should install dependencies", (done) => {
    exec(`cd ${appDirectory} && npm install`, (error) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }
      const packageJson = JSON.parse(fs.readFileSync(`${appDirectory}/package-lock.json`, "utf8"));
      assert.equal(packageJson.name, "demo-electron-app");
      done();
    });
  });

  for (const scenario of scenarios) {
    it(`should complete Electron scenario: ${scenario}`, async () => {
      const { stdout, stderr } = await runElectron(scenario);
      assert(
        stdout.includes(`Scenario completed: ${scenario}`),
        `Check that the ${scenario} scenario reached application teardown`,
      );

      if (diagnosticsEnabled) {
        assertLifecycleTrace(scenario, stderr);
      }

      if (scenario === fullScenario) {
        assert(
          stdout.includes("Created OpenVINO Runtime Core"),
          "Check that openvino-node operates fine",
        );
        assert(
          stdout.includes("Model read successfully: ModelWrap {}"),
          "Check that model is read successfully",
        );
        assert(
          stdout.includes("Infer request result: { fc_out: TensorWrap {} }"),
          "Check that infer request result is successful",
        );
      }
    });
  }

  after((done) => {
    exec(`rm -rf ${appDirectory}`, (error) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }

      done();
    });
  });
});

function assertLifecycleTrace(scenario, stderr) {
  const traces = stderr
    .split("\n")
    .filter((line) => line.startsWith("[OV_NODE_LIFECYCLE]"))
    .map((line) => {
      const match = line.match(/component=(\S+) event=(\S+) context=(\S+)/);
      assert(match, `Parse lifecycle trace line: ${line}`);

      return { component: match[1], event: match[2], context: match[3] };
    });
  const inferRequestTraces = traces.filter(({ component }) => component === "InferRequest");
  const tensorTraces = traces.filter(({ component }) => component === "TensorImpl");

  assert.strictEqual(
    inferRequestTraces.length > 0,
    inferRequestScenarios.has(scenario),
    `Check InferRequest lifecycle trace for scenario ${scenario}`,
  );
  assert.strictEqual(
    tensorTraces.length > 0,
    tensorScenarios.has(scenario),
    `Check TensorImpl lifecycle trace for scenario ${scenario}`,
  );

  if (tensorScenarios.has(scenario)) {
    assertLifecycleEventOrder(scenario, tensorTraces, ["create-context", "add-cleanup-hook"]);
  }

  if (tensorCleanupScenarios.has(scenario)) {
    assertLifecycleEventOrder(scenario, tensorTraces, [
      "cleanup-hook",
      "release-tsfn",
      "tsfn-finalizer",
      "remove-cleanup-hook",
      "tsfn-finalizer-complete",
    ]);
  }
}

function assertLifecycleEventOrder(scenario, traces, expectedEvents) {
  let previousIndex = -1;
  for (const expectedEvent of expectedEvents) {
    const eventIndex = traces.findIndex(
      ({ event }, index) => index > previousIndex && event === expectedEvent,
    );
    assert.notStrictEqual(
      eventIndex,
      -1,
      `Check ${expectedEvent} lifecycle order for scenario ${scenario}`,
    );
    previousIndex = eventIndex;
  }
}

function runElectron(scenario) {
  return new Promise((resolve, reject) => {
    const output = { stdout: [], stderr: [] };
    let outputSequence = 0;
    const electronExecutable = require(path.resolve(appDirectory, "node_modules/electron"));
    const child = spawn(electronExecutable, ["--disable-gpu", "--no-sandbox", "."], {
      cwd: appDirectory,
      env: { ...process.env, OV_ELECTRON_SCENARIO: scenario },
    });

    for (const streamName of ["stdout", "stderr"]) {
      const lineReader = readline.createInterface({ input: child[streamName] });
      lineReader.on("line", (line) => {
        output[streamName].push(line);
        process.stdout.write(
          `[OV_E2E_PARENT] seq=${++outputSequence} scenario=${scenario} stream=${streamName} ${line}\n`,
        );
      });
    }

    child.once("error", reject);
    child.once("close", (exitCode, signal) => {
      process.stdout.write(
        `[OV_E2E_PARENT] scenario=${scenario} exit_code=${exitCode ?? "null"} signal=${signal ?? "none"}\n`,
      );

      const result = {
        stdout: output.stdout.join("\n"),
        stderr: output.stderr.join("\n"),
      };
      if (exitCode !== 0) {
        reject(
          new Error(
            `Electron scenario ${scenario} failed: exit_code=${exitCode ?? "null"}, signal=${signal ?? "none"}`,
          ),
        );

        return;
      }

      resolve(result);
    });
  });
}
