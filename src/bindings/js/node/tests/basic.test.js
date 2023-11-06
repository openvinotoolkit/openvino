// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

const testXml = getModelPath().xml;
const core = new ov.Core();
const model = core.readModelSync(testXml);
const compiledModel = core.compileModelSync(model, 'CPU');
const modelLike = [[model],
  [compiledModel]];

describe('PrePostProcess', () => {

  it('input() ', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input());
  });

  it('input(size_t input_index)', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0));
  });

  it('input(const std::string& tensor_name)', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input('data'));
  });

});

describe('InputInfo', () => {
  it('tensor()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).tensor());
  });

  it('preprocess()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).preprocess());
  });

  it('model()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).model());
  });

  it('tensor(param) throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).tensor(0),
      /Function does not take any parameters./);
  });

  it('preprocess(param) throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).preprocess(0),
      /Function does not take any parameters./);
  });

  it('model(param) throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).model(0),
      /Function does not take any parameters./);
  });

  it('tensor().setElementType()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).tensor().setElementType(ov.element.u8));
  });

  it('tensor().setElementType() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).tensor().setElementType(),
      /Wrong number of parameters./);
  });

  it('tensor().setElementType() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).tensor().setElementType('invalidType'),
      /Cannot create ov::element::Type/);
  });

  it('tensor().SetShape()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).tensor().setShape([1, 10]));
  });

  it('tensor().SetShape() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).tensor().setShape(),
      /Wrong number of parameters./);
  });

  it('tensor().setLayout()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).tensor().setLayout('NHWC'));
  });

  it('tensor().setLayout() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).tensor().setLayout(),
      /Wrong number of parameters./);
  });

  it('preprocess().resize()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).preprocess().resize(ov.resizeAlgorithm.RESIZE_LINEAR));
  });

  it('preprocess().resize() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).preprocess().resize(ov.resizeAlgorithm.RESIZE_LINEAR, 'extraArg'),
      /Wrong number of parameters./);
  });

  it('preprocess().resize() no arg throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).preprocess().resize(),
      /Wrong number of parameters./);
  });

  it('model().setLayout()', () => {
    assert.doesNotThrow(() => new ov.PrePostProcessor(model).input(0).model().setLayout('NCHW'));
  });

  it('model().setLayout() throws', () => {
    assert.throws(() => new ov.PrePostProcessor(model).input(0).model().setLayout('NCHW', 'extraArg'),
      /Wrong number of parameters./);
  });

  it('model().setLayout() throws', () => {
    assert.throws(() =>
      new ov.PrePostProcessor(model).input(0).model().setLayout('invalidLayout')
    );
  });

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

  modelLike.forEach(([obj]) => {
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
