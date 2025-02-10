// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('../..');
const assert = require('assert');
const { describe, it, before, beforeEach } = require('node:test');
const { testModels, isModelAvailable } = require('../utils.js');

describe('ov.preprocess.PrePostProcessor tests', () => {
  const { testModelFP32 } = testModels;
  let core = null;
  let model = null;

  before(async () => {
    await isModelAvailable(testModelFP32);
    core = new ov.Core();
  });

  beforeEach(() => {
    model = core.readModelSync(testModelFP32.xml);
  });

  describe('PrePostProcess', () => {
    it('input() ', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input(),
      );
    });

    it('input(size_t input_index)', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input(0),
      );
    });

    it('input(const std::string& tensor_name)', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input('data'),
      );
    });

    it('output() ', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).output(),
      );
    });

    it('output(size_t output_index)', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).output(0),
      );
    });

    it('output(const std::string& tensor_name)', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).output('fc_out'),
      );
    });
  });

  describe('InputInfo', () => {
    it('tensor()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input(0).tensor(),
      );
    });

    it('preprocess()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input(0).preprocess(),
      );
    });

    it('model()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).input(0).model(),
      );
    });

    it('tensor(param) throws', () => {
      assert.throws(
        () => new ov.preprocess.PrePostProcessor(model).input(0).tensor(0),
        /Function does not take any parameters./,
      );
    });

    it('preprocess(param) throws', () => {
      assert.throws(
        () => new ov.preprocess.PrePostProcessor(model).input(0).preprocess(0),
        /Function does not take any parameters./,
      );
    });

    it('model(param) throws', () => {
      assert.throws(
        () => new ov.preprocess.PrePostProcessor(model).input(0).model(0),
        /Function does not take any parameters./,
      );
    });

    it('tensor().setElementType()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .tensor()
          .setElementType(ov.element.u8),
      );
    });

    it('tensor().setElementType() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .tensor()
            .setElementType(),
        /Wrong number of parameters./,
      );
    });

    it('tensor().setElementType() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .tensor()
            .setElementType('invalidType'),
        /Cannot create ov::element::Type/,
      );
    });

    it('tensor().SetShape()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .tensor()
          .setShape([1, 10]),
      );
    });

    it('tensor().SetShape() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .tensor()
            .setShape(),
        /Wrong number of parameters./,
      );
    });

    it('tensor().setLayout()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .tensor()
          .setLayout('NHWC'),
      );
    });

    it('tensor().setLayout() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .tensor()
            .setLayout(),
        /Wrong number of parameters./,
      );
    });

    it('preprocess().resize()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .preprocess()
          .resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR),
      );
    });

    it('preprocess().resize() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .preprocess()
            .resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR, 'extraArg'),
        /Wrong number of parameters./,
      );
    });

    it('preprocess().resize() no arg throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .preprocess()
            .resize(),
        /Wrong number of parameters./,
      );
    });

    it('model().setLayout()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .model()
          .setLayout('NCHW'),
      );
    });

    it('model().setLayout() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .input(0)
            .model()
            .setLayout('NCHW', 'extraArg'),
        /Wrong number of parameters./,
      );
    });

    it('model().setLayout() throws', () => {
      assert.throws(() =>
        new ov.preprocess.PrePostProcessor(model)
          .input(0)
          .model()
          .setLayout('invalidLayout'),
      );
    });
  });

  describe('OutputInfo', () => {
    it('tensor()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model).output(0).tensor(),
      );
    });

    it('tensor(param) throws', () => {
      assert.throws(
        () => new ov.preprocess.PrePostProcessor(model).output(0).tensor(0),
        /Function does not take any parameters./,
      );
    });

    it('tensor().setElementType()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .output(0)
          .tensor()
          .setElementType(ov.element.u8),
      );
    });

    it('tensor().setElementType() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .output(0)
            .tensor()
            .setElementType(),
        /Wrong number of parameters./,
      );
    });

    it('tensor().setElementType() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .output(0)
            .tensor()
            .setElementType('invalidType'),
        /Cannot create ov::element::Type/,
      );
    });

    it('tensor().setLayout()', () => {
      assert.doesNotThrow(() =>
        new ov.preprocess.PrePostProcessor(model)
          .output(0)
          .tensor()
          .setLayout('NHWC'),
      );
    });

    it('tensor().setLayout() throws', () => {
      assert.throws(
        () =>
          new ov.preprocess.PrePostProcessor(model)
            .output(0)
            .tensor()
            .setLayout(),
        /Wrong number of parameters./,
      );
    });
  });
});
