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
const compiledModel = core.compileModel(model, 'CPU');
const modelLike = [[model],
  [compiledModel]];

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
