const fs = require('node:fs');
const assert = require('node:assert');
const { exec } = require('child_process');

describe('E2E test of installation openvino as electron dependency', function() {
  this.timeout(50000);

  before((done) => {
    exec('cp -r ./tests/e2e/demo-electron-app/ demo-electron-app-project', (error) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }

      done();
    });
  });

  it('should install dependencies', (done) => {
    exec('cd demo-electron-app-project && npm install', (error) => {
      if (error) {
        console.error(`exec error: ${error}`);
        return done(error);
      }
      const packageJson = JSON.parse(fs.readFileSync('demo-electron-app-project/package-lock.json', 'utf8'));
      assert.equal(packageJson.name, 'demo-electron-app');
      done();
    });
  });

  it('should run electron package and verify output', (done) => {
    exec('cd demo-electron-app-project && npm start', (error, stdout) => {
      if (error) {
        console.error(`exec error: ${error}`);

        return done(error);
      }

      assert(
        stdout.includes('Created OpenVINO Runtime Core'),
        'Check that openvino-node operates fine',
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
