// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');

const staticShape = '1, 3, 224, 224';
const dynamicShape = '1, -1, 1..3, 224';

describe('PartialShape', () => {
  it('Should detect static shape', () => {
    const partialShape = new ov.PartialShape(staticShape);

    assert.ok(partialShape.isStatic());
  });
  
  it('Should detect dynamic shape', () => {
    const partialShape = new ov.PartialShape(dynamicShape);

    assert.strictEqual(partialShape.isStatic(), false);
  });

  it('Should return shape as string', () => {
    const partialShape1 = new ov.PartialShape(staticShape);
    const partialShape2 = new ov.PartialShape(dynamicShape);

    assert.strictEqual(partialShape1.toString(), `[1,3,224,224]`);
    assert.strictEqual(partialShape2.toString(), `[1,?,1..3,224]`);
  });
});
