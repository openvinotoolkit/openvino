/* global describe, it, before, after */
const fs = require('node:fs');
const util = require('node:util');
const assert = require('node:assert');
const { exec } = require('child_process');
const execPromise = util.promisify(exec);
const { testModels, downloadTestModel } = require('../unit/utils.js');

describe('E2E testing for OpenVINO as an Electron dependency.', function() {
  this.timeout(50000);

  before(async () => {
    await downloadTestModel(testModels.testModelFP32);
    await execPromise('cp -r ./tests/e2e/demo-electron-app/ demo-electron-app-project');
  });

  it('should install dependencies', (done) => {
    exec('cd demo-electron-app-project && npm install', (error) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }
      const packageJson = JSON.parse(
        fs.readFileSync('demo-electron-app-project/package-lock.json', 'utf8'),
      );
      assert.equal(packageJson.name, 'demo-electron-app');
      done();
    });
  });

  it('should run electron package and verify output', (done) => {
    exec(`cd demo-electron-app-project && npm start`, (error, stdout) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }

      assert(
        stdout.includes('Created OpenVINO Runtime Core'),
        'Check that openvino-node operates fine',
      );
      assert(
        stdout.includes('Model read successfully: ModelWrap {}'),
        'Check that model is read successfully',
      );
      assert(
        stdout.includes('Infer request result: { fc_out: TensorWrap {} }'),
        'Check that infer request result is successful',
      );
      done();
    });
  });

  after((done) => {
    exec('rm -rf demo-electron-app-project', (error) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }

      done();
    });
  });
});
