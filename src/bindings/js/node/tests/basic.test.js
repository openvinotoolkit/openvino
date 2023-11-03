// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

let testXml = getModelPath();
const core = new ov.Core();
const model = core.readModelSync(testXml);
const compiledModel = core.compileModelSync(model, 'CPU');
const modelLike = [[model],
  [compiledModel]];

describe('Core.compileModelSync()', () => {
  const tput = {'PERFORMANCE_HINT': 'THROUGHPUT'};

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
      () => core.compileModelSync(model, 'CPU', {'PERFORMANCE_HINT': tput}),
      /Cannot convert Napi::Value to ov::Any/
    );
  });

  it('compileModelSync(model) throws if the number of arguments is invalid', () => {
    assert.throws(
      () => core.compileModelSync(model),
      /Invalid number of arguments/
    );
  });

} );

describe('Core.compileModel()', () => {
  const tput = {'PERFORMANCE_HINT': 'THROUGHPUT'};

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
      () => core.compileModel(model, 'CPU', {'PERFORMANCE_HINT': tput}).then(),
      /Cannot convert Napi::Value to ov::Any/
    );
  });

  it('compileModel(model) throws if the number of arguments is invalid', () => {
    assert.throws(
      () => core.compileModel(model).then(),
      /Invalid number of arguments/
    );
  });

} );

describe('Output class', () => {

  modelLike.forEach( ([obj]) => {
    it('Output getters and properties', () => {
      assert.strictEqual(typeof obj.output(), 'object');
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
  modelLike.forEach( ([obj]) => {
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
