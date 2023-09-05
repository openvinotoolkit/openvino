// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

function getModelPath(isFP16=false) {
  const basePath = '../../python/tests/';
  return path.join(basePath, 'test_utils', 'utils', `test_model_fp${isFP16 ? 16 : 32}.xml`);
}

module.exports = {
  getModelPath,
};
