// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

function getModelPath(isFP16=false) {
  const basePath = '../../python/tests/';
  let testXml;
  if (isFP16) {
    testXml = path.join(basePath, 'test_utils', 'utils', 'test_model_fp16.xml');
  } else {
    testXml = path.join(basePath, 'test_utils', 'utils', 'test_model_fp32.xml');
  }

  return testXml;
}

module.exports = {
  getModelPath,
};
