// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('..');
const assert = require('assert');
const { describe, it } = require('node:test');

const core = new ov.Core();

it('Core.setProperty()', () => {
  const tmpDir = '/tmp';

  core.setProperty({ 'CACHE_DIR': tmpDir });

  const cacheDir = core.getProperty('CACHE_DIR');

  assert.equal(cacheDir, tmpDir);
});

it('Core.setProperty(\'CPU\')', () => {
  const tmpDir = '/tmp';

  core.setProperty('CPU', { 'CACHE_DIR': tmpDir });

  const cacheDir = core.getProperty('CPU', 'CACHE_DIR');

  assert.equal(cacheDir, tmpDir);
});

it('Core.getProperty(\'CPU\', \'SUPPORTED_PROPERTIES\') is Array', () => {
  const supportedPropertiesArray = core.getProperty('CPU', 'SUPPORTED_PROPERTIES');

  assert.ok(Array.isArray(supportedPropertiesArray));
});

it('Core.setProperty(\'CPU\', { \'NUM_STREAMS\': 5 })', () => {
  const streams = 5;

  core.setProperty('CPU', { 'NUM_STREAMS': streams });
  const result = core.getProperty('CPU', 'NUM_STREAMS');

  assert.equal(result, streams);
});

it('Core.setProperty(\'CPU\', { \'INFERENCE_NUM_THREADS\': 3 })', () => {
  const threads = 3;

  core.setProperty('CPU', { 'INFERENCE_NUM_THREADS': threads });
  const result = core.getProperty('CPU', 'INFERENCE_NUM_THREADS');

  assert.equal(result, threads);
});

it('Core.addExtension() with empty parameters', () => {
  assert.throws(
    () => core.addExtension(),
    /addExtension method applies one argument of string type/
  );
});

it('Core.addExtension(\'not_exists\') with non-existed library', () => {
  const notExistsExt = 'not_exists';

  assert.throws(
    () => core.addExtension(notExistsExt),
    /Cannot load library 'not_exists'/
  );
});
