// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

module.exports = { getModelPath };

function getModelPath(isFP16=false) {
  const basePath = 'tests/test_models/';
  const modelName = `test_model_fp${isFP16 ? 16 : 32}`;

  return {
    xml: path.join(basePath, `${modelName}.xml`),
    bin: path.join(basePath, `${modelName}.bin`),
  };
}
