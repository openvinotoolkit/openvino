// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const fs = require('node:fs');
const { addon: ov } = require('../..');
const assert = require('assert');
const { describe, it, before, beforeEach } = require('node:test');
const { testModels, isModelAvailable } = require('../utils.js');

describe('Tests for reading model.', () => {
  const { testModelFP32 } = testModels;
  let modelFile = null;
  let modelStr = null;
  let weightsFile = null;
  let weightsTensor = null;
  let core = null;

  before(async () => {
    await isModelAvailable(testModelFP32);
    modelFile = fs.readFileSync(testModelFP32.xml);
    modelStr = fs.readFileSync(testModelFP32.xml, 'utf8');
    weightsFile = fs.readFileSync(testModelFP32.bin);
  });

  beforeEach(() => {
    core = new ov.Core();
    weightsTensor = new ov.Tensor(
      ov.element.u8,
      [weightsFile.buffer.byteLength],
      new Uint8Array(weightsFile.buffer),
    );
  });

  describe('Core.readModeSync', () => {
    it('readModeSync(xmlPath) ', () => {
      const model = core.readModelSync(testModelFP32.xml);
      assert.ok(model instanceof ov.Model);
      assert.equal(model.inputs.length, 1);
    });

    it('readModeSync(xmlPath, weightsPath) ', () => {
      const model = core.readModelSync(testModelFP32.xml, testModelFP32.bin);
      assert.ok(model instanceof ov.Model);
      assert.equal(model.inputs.length, 1);
    });

    it('readModeSync throws', () => {
      assert.throws(
        () => core.readModelSync(core),
        /'readModelSync' method called with incorrect parameters./,
      );
    });

    it('readModelSync(modelString, weightsTensor) ', () => {
      const model = core.readModelSync(modelStr, weightsTensor);
      assert.ok(model instanceof ov.Model);
      assert.equal(model.inputs.length, 1);
    });

    it('readModelSync(modelUint8ArrayBuffer, weightsUint8ArrayBuffer) ', () => {
      const model = core.readModelSync(
        new Uint8Array(modelFile.buffer),
        new Uint8Array(weightsFile.buffer),
      );
      assert.ok(model instanceof ov.Model);
      assert.equal(model.inputs.length, 1);
    });
  });

  describe('Core.readModel', () => {
    it('readModel(xmlPath) ', async () => {
      const model = await core.readModel(testModelFP32.xml);
      assert.equal(model.inputs.length, 1);
    });

    it('readModel(xmlPath, weightsPath) ', async () => {
      const model = await core.readModel(testModelFP32.xml, testModelFP32.bin);
      assert.equal(model.inputs.length, 1);
    });

    it('readModel(modelString, weightsTensor) ', async () => {
      const model = await core.readModel(modelStr, weightsTensor);
      assert.ok(model instanceof ov.Model);
      assert.equal(model.inputs.length, 1);
    });

    it('readModel(Uint8ArrayBuffer, Uint8ArrayBuffer) ', async () => {
      const model = await core.readModel(
        new Uint8Array(modelFile.buffer),
        new Uint8Array(weightsFile.buffer),
      );
      assert.equal(model.inputs.length, 1);
    });
  });
});
