// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('..');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

const testXml = getModelPath().xml;
const core = new ov.Core();
const model = core.readModelSync(testXml);
const compiledModel = core.compileModelSync(model, 'CPU');
const modelLike = [[model],
  [compiledModel]];

it('Core.getAvailableDevices()', () => {    
    const devices = core.getAvailableDevices();
    
    assert.ok(devices.includes('CPU'));
});

describe('Core.getVersions()', () => {

  it('getVersions(validDeviceName: string)', () => {    
    const deviceVersion = core.getVersions('CPU');
    assert.strictEqual(typeof deviceVersion, 'object');
    assert.strictEqual(typeof deviceVersion.CPU, 'object');
    assert.strictEqual(typeof deviceVersion.CPU.buildNumber, 'string');
    assert.strictEqual(typeof deviceVersion.CPU.description, 'string');
  });

  it('getVersions() throws if no arguments are passed into the function', () => {
    assert.throws(
      () => core.getVersions(),
      {message: 'getVersions() method expects 1 argument of string type.'}
    );
  });

  it('getVersions() throws if non string coercable arguments are passed into the function', () => {
    assert.throws(
      () => core.getVersions({ deviceName: 'CPU' }),
      {message: 'The argument in getVersions() method must be a string or convertible to a string.'}
    );
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
  const tput = { 'PERFORMANCE_HINT': 'THROUGHPUT' };

  it('compileModelSync(model:Model, deviceName: string, config: {}) ', () => {
    const cm = core.compileModelSync(model, 'CPU', tput);
    assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
  });

  it('compileModelSync(model:model_path, deviceName: string, config: {}) ', () => {
    const cm = core.compileModelSync(testXml, 'CPU', tput);
    assert.equal(cm.inputs.length, 1);
  });

  it('compileModelSync(model:model_path, deviceName: string) ', () => {
    const cm = core.compileModelSync(testXml, 'CPU');
    assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
  });

  it('compileModelSync(model, device, config) throws when config is a string', () => {
    assert.throws(
      () => core.compileModelSync(model, 'CPU', 'string'),
      /Cannot convert Napi::Value to std::map<std::string, ov::Any>/
    );
  });

  it('compileModelSync(model, device, config) throws when config value is not a string', () => {
    assert.throws(
      () => core.compileModelSync(model, 'CPU', { 'PERFORMANCE_HINT': tput }),
      /Cannot convert Napi::Value to ov::Any/
    );
  });

  it('compileModelSync(model) throws if the number of arguments is invalid', () => {
    assert.throws(
      () => core.compileModelSync(model),
      /Invalid number of arguments/
    );
  });

});

describe('Core.compileModel()', () => {
  const tput = { 'PERFORMANCE_HINT': 'THROUGHPUT' };

  it('compileModel(model:Model, deviceName: string, config: {}) ', () => {
    core.compileModel(model, 'CPU', tput).then(cm => {
      assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
    });

  });

  it('compileModel(model:model_path, deviceName: string, config: {}) ', () => {
    core.compileModel(testXml, 'CPU', tput).then(cm => {
      assert.equal(cm.inputs.length, 1);
    });

  });

  it('compileModel(model:model_path, deviceName: string) ', () => {
    core.compileModel(testXml, 'CPU').then(cm => {
      assert.deepStrictEqual(cm.output(0).shape, [1, 10]);
    });

  });

  it('compileModel(model, device, config) throws when config isn\'t an object', () => {
    assert.throws(
      () => core.compileModel(model, 'CPU', 'string').then(),
      /Cannot convert Napi::Value to std::map<std::string, ov::Any>/
    );
  });

  it('compileModel(model, device, config) throws when config value is not a string', () => {
    assert.throws(
      () => core.compileModel(model, 'CPU', { 'PERFORMANCE_HINT': tput }).then(),
      /Cannot convert Napi::Value to ov::Any/
    );
  });

  it('compileModel(model) throws if the number of arguments is invalid', () => {
    assert.throws(
      () => core.compileModel(model).then(),
      /Invalid number of arguments/
    );
  });

});

describe('Output class', () => {

  it('Output type', () => {
    assert.ok(model.output() instanceof ov.Output);
  });

  it('ConstOutput type', () => {
    assert.ok(compiledModel.output() instanceof ov.ConstOutput);
  });

  modelLike.forEach(([obj]) => {
    it('Output getters and properties', () => {
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

describe('Input class for ov::Input<const ov::Node>', () => {

  it('Output type', () => {
    assert.ok(model.input() instanceof ov.Output);
  });

  it('ConstOutput type', () => {
    assert.ok(compiledModel.input() instanceof ov.ConstOutput);
  });

  modelLike.forEach(([obj]) => {
    it('input() is typeof object', () => {
      assert.strictEqual(typeof obj.input(), 'object');
    });

    it('inputs property', () => {
      assert.strictEqual(obj.inputs.length, 1);
    });

    it('input().toString()', () => {
      assert.strictEqual(obj.input().toString(), 'data');
    });

    it('input(idx: number).ToString() method', () => {
      assert.strictEqual(obj.input(0).toString(), 'data');
    });

    it('input(tensorName: string).ToString() method', () => {
      assert.strictEqual(obj.input('data').toString(), 'data');
    });

    it('input().getAnyName() and anyName', () => {
      assert.strictEqual(obj.input().getAnyName(), 'data');
      assert.strictEqual(obj.input().anyName, 'data');
    });

    it('input(idx).shape property with dimensions', () => {
      assert.deepStrictEqual(obj.input(0).shape, [1, 3, 32, 32]);
      assert.deepStrictEqual(obj.input(0).getShape(), [1, 3, 32, 32]);
    });
  });

});

it('Test exportModel()/importModel()', () => {
  const userStream = compiledModel.exportModelSync();
  const newCompiled = core.importModelSync(userStream, 'CPU');
  const epsilon = 0.5;
  const tensor = Float32Array.from({ length: 3072 }, () => (Math.random() + epsilon));

  const inferRequest = compiledModel.createInferRequest();
  const res1 = inferRequest.infer([tensor]);
  const newInferRequest = newCompiled.createInferRequest();
  const res2 = newInferRequest.infer([tensor]);

  assert.deepStrictEqual(res1['fc_out'].data[0], res2['fc_out'].data[0]);
});
