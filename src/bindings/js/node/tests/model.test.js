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

describe('Node.js Model.isDynamic()', () => {

  it('should return a boolean value indicating if the model is dynamic', () => {
    const result = model.isDynamic();
    assert.strictEqual(typeof result, 'boolean', 'isDynamic() should return a boolean value');
  });

  it('should not accept any arguments', () => {
    assert.throws(() => {
      model.isDynamic('unexpected argument');
    }, /^Error: isDynamic\(\) does not accept any arguments\.$/, 'Expected isDynamic to throw an error when called with arguments');
  });

  it('returns false for a static model', () => {
    const expectedStatus = false;
    assert.strictEqual(model.isDynamic(), expectedStatus, 'Expected isDynamic to return false for a static model');
  });
});
