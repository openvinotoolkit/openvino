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
    assert.strictEqual(
      typeof result,
      'boolean',
      'isDynamic() should return a boolean value'
    );
  });

  it('should not accept any arguments', () => {
    assert.throws(
      () => {
        model.isDynamic('unexpected argument');
      },
      /^Error: isDynamic\(\) does not accept any arguments\.$/,
      'Expected isDynamic to throw an error when called with arguments'
    );
  });

  it('returns false for a static model', () => {
    const expectedStatus = false;
    assert.strictEqual(
      model.isDynamic(),
      expectedStatus,
      'Expected isDynamic to return false for a static model'
    );
  });
});

describe('Node.js getFriendlyName() / setFriendlyName()', () => {
  describe('getFriendlyName()', () => {
    it('returns the unique name of the model if no friendly name is set', () => {
      const expectedName = 'test_model';
      assert.strictEqual(model.getFriendlyName(), expectedName);
    });
    it('throws an error when called with arguments', () => {
      assert.throws(
        () => model.getFriendlyName('unexpected argument'),
        /getFriendlyName\(\) does not take any arguments/
      );
    });
  });
  describe('setFriendlyName()', () => {
    it('sets a friendly name for the model', () => {
      assert.doesNotThrow(() => model.setFriendlyName('MyFriendlyName'));
    });

    it('throws an error when called without a string argument', () => {
      assert.throws(
        () => model.setFriendlyName(),
        /Expected a single string argument for the friendly name/
      );
      assert.throws(
        () => model.setFriendlyName(123),
        /Expected a single string argument for the friendly name/
      );
    });

    it('throws an error when called with multiple arguments', () => {
      assert.throws(
        () => model.setFriendlyName('Name1', 'Name2'),
        /Expected a single string argument for the friendly name/
      );
    });

    it('returns the set friendly name of the model', () => {
      const friendlyName = 'MyFriendlyModel';
      model.setFriendlyName(friendlyName);
      assert.strictEqual(model.getFriendlyName(), friendlyName);
    });

    it('retains the last set friendly name when set multiple times', () => {
      model.setFriendlyName('InitialName');
      model.setFriendlyName('FinalName');
      assert.strictEqual(model.getFriendlyName(), 'FinalName');
    });

    it('handles setting an empty string as a friendly name', () => {
      assert.doesNotThrow(() => model.setFriendlyName(''));
      assert.strictEqual(model.getFriendlyName(), 'Model1');
    });
  });
});

describe('Model.getOutputSize()', () => {

  it('should return a number indicating number of outputs for the model', () => {
    const result = model.getOutputSize();
    assert.strictEqual(typeof result, 'number', 'getOutputSize() should return a number');
  });

  it('should not accept any arguments', () => {
    assert.throws(() => {
      model.getOutputSize('unexpected argument');
    }, /^Error: getOutputSize\(\) does not accept any arguments\.$/, 'Expected getOutputSize to throw an error when called with arguments');
  });

  it('should return 1 for the default model', () => {
    assert.strictEqual(model.getOutputSize(), 1, 'Expected getOutputSize to return 1 for the default model');
  });
});

describe('Model.getOutputElementType()', () => {
  it('should return a string for the element type ', () => {
    const result = model.getOutputElementType(0);
    assert.strictEqual(typeof result, 'string',
      'getOutputElementType() should return a string');
  });

  it('should accept a single integer argument', () => {
    assert.throws(() => {
      model.getOutputElementType();
    }, /^Error: getOutputElementType method called with incorrect parameters\./,
     'Should throw when called without arguments');

    assert.throws(() => {
      model.getOutputElementType('unexpected argument');
    }, /^Error: getOutputElementType method called with incorrect parameters\./,
    'Should throw on non-number argument');

    assert.throws(() => {
      model.getOutputElementType(0, 1);
    }, /^Error: getOutputElementType method called with incorrect parameters\./,
    'Should throw on multiple arguments');

    assert.throws(() => {
      model.getOutputElementType(3.14);
    }, /^Error: getOutputElementType method called with incorrect parameters\./,
    'Should throw on non-integer number');
  });

  it('should return a valid element type for the default model', () => {
    const elementType = model.getOutputElementType(0);
    assert.ok(typeof elementType === 'string' && elementType.length > 0,
      `Expected a non-empty string, got ${elementType}`);
  });

  it('should throw an error for out-of-range index', () => {
    const outputSize = model.getOutputSize();
    assert.throws(
      () => { model.getOutputElementType(outputSize); },
      /^Error:/,
      'Should throw for out-of-range index'
    );
  });
});