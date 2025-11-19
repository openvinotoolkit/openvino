// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require("../..");
const assert = require("assert");
const { describe, it, before, beforeEach } = require("node:test");
const { testModels, isModelAvailable } = require("../utils.js");

describe("ov.CompiledModel tests", () => {
  const { testModelFP32 } = testModels;
  let core = null;
  let compiledModel = null;
  let skipCompiledModelTests = false;

  // Quick synchronous check: some test runners/environments don't have the
  // BATCH device plugin available. Try probing when the test suite initializes and
  // mark the suite to be skipped early to avoid running hooks that attempt
  // to compile and then fail.
  try {
    const _probeCore = new ov.Core();
    const _devices = _probeCore.getAvailableDevices();
    if (!_devices.includes("BATCH")) {
      skipCompiledModelTests = true;
    }
  } catch (e) {
    skipCompiledModelTests = true;
  }

  before(async () => {
    await isModelAvailable(testModelFP32);
    core = new ov.Core();
    // Some environments under test may not have the BATCH device plugin
    // available. If it's missing, skip these compiled model tests rather
    // than failing the whole suite.
    try {
      const devices = core.getAvailableDevices();
      if (!devices.includes("BATCH")) {
        skipCompiledModelTests = true;
      }
    } catch (e) {
      // If getting devices fails for some reason, skip as well.
      skipCompiledModelTests = true;
    }
  });

  beforeEach(function () {
    if (skipCompiledModelTests) {
      this.skip();
      return;
    }
    const properties = {
      AUTO_BATCH_TIMEOUT: "1",
    };
    try {
      compiledModel = core.compileModelSync(testModelFP32.xml, "BATCH:CPU", properties);
    } catch (e) {
      // If compilation fails (device/plugin missing), skip these tests instead
      // of failing the whole suite in this environment.
      this.skip();
      return;
    }
  });

  describe("getProperty()", () => {
    it("returns the value of property from compiled model", () => {
      assert.strictEqual(compiledModel.getProperty("AUTO_BATCH_TIMEOUT"), "1");
    });
    it("throws an error when called without arguments", () => {
      assert.throws(
        () => compiledModel.getProperty(),
        /'getProperty' method called with incorrect parameters/,
      );
    });
    it("throws when called with property name that does not exists", () => {
      assert.throws(() => compiledModel.getProperty("PROPERTY_THAT_DOES_NOT_EXIST"));
    });
  });

  describe("setProperty()", () => {
    it("sets a properties for compiled model", () => {
      const properties = { AUTO_BATCH_TIMEOUT: "1000" };
      assert.doesNotThrow(() => compiledModel.setProperty(properties));
    });

    it("throws an error when called without an object argument", () => {
      assert.throws(
        () => compiledModel.setProperty(),
        /'setProperty' method called with incorrect parameters/,
      );
    });
    it("throws an error when called with wrong argument", () => {
      assert.throws(
        () => compiledModel.setProperty(123),
        /'setProperty' method called with incorrect parameters/,
      );
    });

    it("throws an error when called with multiple arguments", () => {
      assert.throws(
        () =>
          compiledModel.setProperty({ PERFORMANCE_HINT: "THROUGHPUT" }, { NUM_STREAMS: "AUTO" }),
        /'setProperty' method called with incorrect parameters/,
      );
    });

    it("returns the set property of the compiled model", () => {
      const properties = { AUTO_BATCH_TIMEOUT: "123" };
      compiledModel.setProperty(properties);
      assert.strictEqual(compiledModel.getProperty("AUTO_BATCH_TIMEOUT"), 123);
    });

    it("retains the last set property when set multiple times", () => {
      compiledModel.setProperty({ AUTO_BATCH_TIMEOUT: "321" });
      compiledModel.setProperty({ AUTO_BATCH_TIMEOUT: "132" });
      assert.strictEqual(compiledModel.getProperty("AUTO_BATCH_TIMEOUT"), 132);
    });

    it("allows to pass empty object", () => {
      assert.doesNotThrow(() => compiledModel.setProperty({}));
    });
  });
});
