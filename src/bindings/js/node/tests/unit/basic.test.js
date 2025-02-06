// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('../..');
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { after, describe, it, before, beforeEach } = require('node:test');
const {
  testModels,
  compareModels,
  isModelAvailable,
  sleep,
  lengthFromShape,
} = require('../utils.js');
const epsilon = 0.5;

describe('ov basic tests.', () => {
  const { testModelFP32 } = testModels;
  let core = null;
  let model = null;
  let compiledModel = null;
  let modelLike = null;
  let outDir = null;

  before(async () => {
    outDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'ov_js_out_'));
    await isModelAvailable(testModelFP32);
  });

  beforeEach(() => {
    core = new ov.Core();
    model = core.readModelSync(testModelFP32.xml);
    compiledModel = core.compileModelSync(model, 'CPU');
    modelLike = [model, compiledModel];
  });

  after(async () => {
    // Wait to ensure the model file is released
    await sleep(1);
    await fs.promises.rm(outDir, { recursive: true });
  });

  it('Core.getAvailableDevices()', () => {
    const devices = core.getAvailableDevices();

    assert.ok(devices.includes('CPU'));
  });

  describe('ov.saveModelSync()', () => {
    it('saveModelSync(model, path, compressToFp16=true)', () => {
      const xmlPath = path.join(outDir, `${model.getName()}_fp16.xml`);
      assert.doesNotThrow(() => ov.saveModelSync(model, xmlPath, true));

      const savedModel = core.readModelSync(xmlPath);
      assert.doesNotThrow(() => compareModels(model, savedModel));
    });

    it('saveModelSync(model, path, compressToFp16)', () => {
      const xmlPath = path.join(outDir, `${model.getName()}_fp32.xml`);
      assert.doesNotThrow(() => ov.saveModelSync(model, xmlPath));

      const savedModel = core.readModelSync(xmlPath);
      assert.doesNotThrow(() => compareModels(model, savedModel));
    });
    it('saveModelSync(model, path, compressToFp16=false)', () => {
      const xmlPath = path.join(outDir, `${model.getName()}_fp32.xml`);
      assert.doesNotThrow(() => ov.saveModelSync(model, xmlPath, false));

      const savedModel = core.readModelSync(xmlPath);
      assert.doesNotThrow(() => compareModels(model, savedModel));
    });

    it('saveModelSync(model) throws', () => {
      const expectedMsg = (
        '\'saveModelSync\' method called with incorrect parameters.\n' +
        'Provided signature: (object) \n' +
        'Allowed signatures:\n' +
        '- (Model, string)\n' +
        '- (Model, string, boolean)\n'
      ).replace(/[()]/g, '\\$&');

      assert.throws(
        () => ov.saveModelSync(model),
        new RegExp(expectedMsg));
    });

    it('saveModelSync(model, path) throws with incorrect path', () => {
      const expectedMsg = (
        'Path for xml file doesn\'t ' +
        'contains file name with \'xml\' extension'
      ).replace(/[()]/g, '\\$&');

      const noXmlPath = `${outDir}${model.getName()}_fp32`;
      assert.throws(
        () => ov.saveModelSync(model, noXmlPath),
        new RegExp(expectedMsg));
    });
  });

  describe('Core.getVersions()', () => {
    it('getVersions(validDeviceName: string)', () => {
      const deviceVersion = core.getVersions('CPU');
      assert.strictEqual(typeof deviceVersion, 'object');
      assert.strictEqual(typeof deviceVersion.CPU, 'object');
      assert.strictEqual(typeof deviceVersion.CPU.buildNumber, 'string');
      assert.strictEqual(typeof deviceVersion.CPU.description, 'string');
    });

    it('getVersions() throws if no arguments are passed', () => {
      assert.throws(() => core.getVersions(), {
        message: 'getVersions() method expects 1 argument of string type.',
      });
    });

    it('getVersions() throws with non string coercable arg.', () => {
      assert.throws(() => core.getVersions({ deviceName: 'CPU' }));
    });
  });

  it('CompiledModel type', () => {
    assert.ok(compiledModel instanceof ov.CompiledModel);
  });

  it('compileModel.createInferRequest()', () => {
    const ir = compiledModel.createInferRequest();
    assert.ok(ir instanceof ov.InferRequest);
  });

  describe('Core.compileModelSync()', () => {
    const tput = { PERFORMANCE_HINT: 'THROUGHPUT' };

    it('compileModelSync(model:Model, deviceName: string, config: {}) ', () => {
      const cm = core.compileModelSync(model, 'CPU', tput);
      assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
    });

    it('compileModelSync(model_path, deviceName, config: {}) ', () => {
      const cm = core.compileModelSync(testModelFP32.xml, 'CPU', tput);
      assert.equal(cm.inputs.length, 1);
    });

    it('compileModelSync(model:model_path, deviceName: string) ', () => {
      const cm = core.compileModelSync(testModelFP32.xml, 'CPU');
      assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
    });

    it('compileModelSync(model, device, configOfTypeString) throws', () => {
      const expectedMsg = (
        '\'compileModelSync\' method called with incorrect parameters.\n' +
        'Provided signature: (object, string, string) \n' +
        'Allowed signatures:\n' +
        '- (string, string)\n' +
        '- (Model, string)\n' +
        '- (string, string, object)\n' +
        '- (Model, string, object)\n'
      ).replace(/[()]/g, '\\$&');

      assert.throws(
        () => core.compileModelSync(model, 'CPU', 'string'),
        new RegExp(expectedMsg),
      );
    });

    it('compileModelSync(model, device, invalidConfig) throws', () => {
      assert.throws(
        () => core.compileModelSync(model, 'CPU', { PERFORMANCE_HINT: tput }),
        /Cannot convert to ov::Any/,
      );
    });

    it('compileModelSync(model) throws with invalid number of args', () => {
      const expectedMsg = (
        '\'compileModelSync\' method called with incorrect parameters.\n' +
        'Provided signature: (object) \n' +
        'Allowed signatures:\n' +
        '- (string, string)\n' +
        '- (Model, string)\n' +
        '- (string, string, object)\n' +
        '- (Model, string, object)\n'
      ).replace(/[()]/g, '\\$&');

      assert.throws(
        () => core.compileModelSync(model),
        new RegExp(expectedMsg),
      );
    });
  });

  describe('Core.compileModel()', () => {
    const tput = { PERFORMANCE_HINT: 'THROUGHPUT' };

    it('compileModel(model:Model, deviceName: string, config: {}) ', () => {
      core.compileModel(model, 'CPU', tput).then((cm) => {
        assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
      });
    });

    it('compileModel(model_path, deviceName, config: {}) ', () => {
      core.compileModel(testModelFP32.xml, 'CPU', tput).then((cm) => {
        assert.equal(cm.inputs.length, 1);
      });
    });

    it('compileModel(model:model_path, deviceName: string) ', () => {
      core.compileModel(testModelFP32.xml, 'CPU').then((cm) => {
        assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
      });
    });

    it('compileModel(model, device, invalidconfig) throws', () => {
      assert.throws(
        () => core.compileModel(model, 'CPU', 'string').then(),
        'Argument #3 must be an Object.',
      );
    });

    it('compileModel(model, device, invalidConfig) throws', () => {
      assert.throws(
        () =>
          core.compileModel(model, 'CPU', { PERFORMANCE_HINT: tput }).then(),
        /Cannot convert to ov::Any/,
      );
    });

    it('compileModel(model) throws with invalid number of args', () => {
      assert.throws(
        () => core.compileModel(model).then(),
        /Invalid number of arguments/,
      );
    });
  });

  describe('ov.Model/ov.CompiledModel output()', () => {
    it('model.output() type', () => {
      assert.ok(model.output() instanceof ov.Output);
    });

    it('compiledModel.output() type', () => {
      assert.ok(compiledModel.output() instanceof ov.ConstOutput);
    });

    it('output() methods/properties', () => {
      modelLike.forEach((obj) => {
        assert.strictEqual(obj.outputs.length, 1);
        // tests for an obj with one output
        assert.strictEqual(obj.output().toString(), 'fc_out');
        assert.strictEqual(obj.output(0).toString(), 'fc_out');
        assert.strictEqual(obj.output('fc_out').toString(), 'fc_out');
        assert.deepStrictEqual(obj.output(0).shape, [1, 10]);
        assert.deepStrictEqual(obj.output(0).getShape(), [1, 10]);
        assert.strictEqual(obj.output().getAnyName(), 'fc_out');
        assert.strictEqual(obj.output().anyName, 'fc_out');
      });
    });
  });

  describe('ov.Model/ov.CompiledModel input()', () => {
    it('ov.Model.input() type', () => {
      assert.ok(model.input() instanceof ov.Output);
    });

    it('ov.CompiledModel.input() type', () => {
      assert.ok(compiledModel.input() instanceof ov.ConstOutput);
    });

    it('input() methods/properties', () => {
      modelLike.forEach((obj) => {
        assert.strictEqual(typeof obj.input(), 'object');
        assert.strictEqual(obj.inputs.length, 1);
        assert.strictEqual(obj.input().toString(), 'data');
        assert.strictEqual(obj.input(0).toString(), 'data');
        assert.strictEqual(obj.input('data').toString(), 'data');
        assert.strictEqual(obj.input().getAnyName(), 'data');
        assert.strictEqual(obj.input().anyName, 'data');

        assert.deepStrictEqual(obj.input(0).shape, testModelFP32.inputShape);
        assert.deepStrictEqual(
          obj.input(0).getShape(),
          testModelFP32.inputShape,
        );
      });
    });
  });

  describe('Test exportModel()/importModel()', () => {
    let tensor = null;
    let userStream = null;
    let res1 = null;

    before(() => {
      tensor = Float32Array.from(
        { length: lengthFromShape(testModelFP32.inputShape) },
        () => Math.random() + epsilon,
      );
      const core = new ov.Core();
      const model = core.readModelSync(testModelFP32.xml);
      const compiledModel = core.compileModelSync(model, 'CPU');
      userStream = compiledModel.exportModelSync();
      const inferRequest = compiledModel.createInferRequest();
      res1 = inferRequest.infer([tensor]);
    });

    it('Test importModelSync(stream, device)', () => {
      const newCompiled = core.importModelSync(userStream, 'CPU');
      const newInferRequest = newCompiled.createInferRequest();
      const res2 = newInferRequest.infer([tensor]);

      assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
    });

    it('Test importModelSync(stream, device, config)', () => {
      const newCompiled = core.importModelSync(userStream, 'CPU', {
        NUM_STREAMS: 1,
      });
      const newInferRequest = newCompiled.createInferRequest();
      const res2 = newInferRequest.infer([tensor]);

      assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
    });

    it('Test importModelSync(stream, device) throws', () => {
      assert.throws(
        () => core.importModelSync(epsilon, 'CPU'),
        /The first argument must be of type Buffer./,
      );
    });

    it('Test importModelSync(stream, device) throws', () => {
      assert.throws(
        () => core.importModelSync(userStream, tensor),
        /The second argument must be of type String./,
      );
    });
    it('Test importModelSync(stream, device, config: tensor) throws', () => {
      assert.throws(
        () => core.importModelSync(userStream, 'CPU', tensor),
        /NotFound: Unsupported property 0 by CPU plugin./,
      );
    });

    it('Test importModelSync(stream, device, config: string) throws', () => {
      const testString = 'test';
      assert.throws(
        () => core.importModelSync(userStream, 'CPU', testString),
        /Passed Napi::Value must be an object./,
      );
    });

    it('Test importModelSync(stream, device, config: unsupported property) \
    throws', () => {
      const tmpDir = '/tmp';
      assert.throws(
        () => core.importModelSync(userStream, 'CPU', { CACHE_DIR: tmpDir }),
        /Unsupported property CACHE_DIR by CPU plugin./,
      );
    });

    it('Test importModel(stream, device)', () => {
      core.importModel(userStream, 'CPU').then((newCompiled) => {
        const newInferRequest = newCompiled.createInferRequest();
        const res2 = newInferRequest.infer([tensor]);
        assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
      });
    });

    it('Test importModel(stream, device, config)', () => {
      core
        .importModel(userStream, 'CPU', { NUM_STREAMS: 1 })
        .then((newCompiled) => {
          const newInferRequest = newCompiled.createInferRequest();
          const res2 = newInferRequest.infer([tensor]);

          assert.deepStrictEqual(
            res1['fc_out'].data[0],
            res2['fc_out'].data[0],
          );
        });
    });

    it('Test importModel(stream, device) throws', () => {
      assert.throws(
        () => core.importModel(epsilon, 'CPU').then(),
        /'importModel' method called with incorrect parameters./,
      );
    });

    it('Test importModel(stream, device) throws', () => {
      assert.throws(
        () => core.importModel(userStream, tensor).then(),
        /'importModel' method called with incorrect parameters./,
      );
    });

    it('Test importModel(stream, device, config: string) throws', () => {
      const testString = 'test';
      assert.throws(
        () => core.importModel(userStream, 'CPU', testString).then(),
        /'importModel' method called with incorrect parameters./,
      );
    });
  });
});
