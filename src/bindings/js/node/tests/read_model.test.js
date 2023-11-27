// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const fs = require('node:fs');
const ov = require('../bin/ov_node_addon.node');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

const { xml: modelPath, bin: weightsPath } = getModelPath();
const modelFile = fs.readFileSync(modelPath);
const weightsFile = fs.readFileSync(weightsPath);

const core = new ov.Core();

describe('Core.readModeSync', () => {
  it('readModeSync(xmlPath) ', () => {
    const model = core.readModelSync(modelPath);
    assert.equal(model.inputs.length, 1);
  });

  it('readModeSync(xmlPath, weightsPath) ', () => {
    const model = core.readModelSync(modelPath, weightsPath);
    assert.equal(model.inputs.length, 1);
  });

  it('readModeSync(modelUint8ArrayBuffer, weightsUint8ArrayBuffer) ', () => {
    const model = core.readModelSync(
      new Uint8Array(modelFile.buffer),
      new Uint8Array(weightsFile.buffer),
    );
    assert.equal(model.inputs.length, 1);
  });
});

describe('Core.readModel', () => {
  it('readModel(xmlPath) ', async () => {
    const model = await core.readModelSync(modelPath);
    assert.equal(model.inputs.length, 1);
  });

  it('readModel(xmlPath, weightsPath) ', async () => {
    const model = await core.readModelSync(modelPath, weightsPath);
    assert.equal(model.inputs.length, 1);
  });

  it('readModeSync(modelUint8ArrayBuffer, weightsUint8ArrayBuffer) ', async () => {
    const model = await core.readModelSync(
      new Uint8Array(modelFile.buffer),
      new Uint8Array(weightsFile.buffer),
    );
    assert.equal(model.inputs.length, 1);
  });
});
