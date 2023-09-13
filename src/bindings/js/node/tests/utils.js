// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

function getModelPath(isFP16=false) {
  const basePath = 'tests/test_models/';

  return path.join(basePath, `test_model_fp${isFP16 ? 16 : 32}.xml`);
}

module.exports = {
  getModelPath,
};
