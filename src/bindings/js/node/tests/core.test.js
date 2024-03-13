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
  const tmpDir = '/home/nvishnya/tmp123';

  core.setProperty('CPU', { 'CACHE_DIR': tmpDir });

  const cacheDir = core.getProperty('CPU', 'CACHE_DIR');

  assert.equal(cacheDir, tmpDir);
});

it('Core.getProperty(\'CPU\', \'SUPPORTED_PROPERTIES\') is Array', () => {    
  const tmpDir = '/home/nvishnya/tmp123';

  const supportedPropertiesArray = core.getProperty('CPU', 'SUPPORTED_PROPERTIES');

  assert.ok(Array.isArray(supportedPropertiesArray), tmpDir);
});
