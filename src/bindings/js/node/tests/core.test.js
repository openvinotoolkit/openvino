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
