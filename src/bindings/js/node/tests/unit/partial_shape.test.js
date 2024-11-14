// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('../..');
const assert = require('assert');
const { describe, it } = require('node:test');

const staticShape = '1, 3, 224, 224';
const dynamicShape = '?, -1, 1..3, 224';

describe('ov.PartialShape tests', () => {
  it('allows create empty shape', () => {
    const partialShape = new ov.PartialShape();

    assert.strictEqual(partialShape.toString(), '[]');
  });

  it('should detect static shape', () => {
    const partialShape = new ov.PartialShape(staticShape);

    assert.ok(partialShape.isStatic());
  });

  it('should detect dynamic shape', () => {
    const partialShape = new ov.PartialShape(dynamicShape);

    assert.strictEqual(partialShape.isStatic(), false);
  });

  it('should return shape as string for static shape', () => {
    const partialShape = new ov.PartialShape(staticShape);

    assert.strictEqual(partialShape.toString(), '[1,3,224,224]');
  });

  it('should return shape as string for dynamic shape', () => {
    const partialShape = new ov.PartialShape(dynamicShape);

    assert.strictEqual(partialShape.toString(), '[?,?,1..3,224]');
  });

  it('should return array with dimensions for dynamic shape', () => {
    const partialShape = new ov.PartialShape(staticShape);

    assert.deepStrictEqual(partialShape.getDimensions(), [1, 3, 224, 224]);
  });

  it('should return array with dimensions for dynamic shape', () => {
    const partialShape = new ov.PartialShape(dynamicShape);

    assert.deepStrictEqual(partialShape.getDimensions(), [-1, -1, [1, 3], 224]);
  });
});
