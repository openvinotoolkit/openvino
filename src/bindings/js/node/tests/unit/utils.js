// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');
const fs = require('node:fs/promises');
const {
  downloadFile,
  checkIfPathExists,
} = require('../../scripts/download_runtime');

const modelDir = 'tests/unit/test_models/';
const testModels = {
  testModelFP32: {
    xml: 'test_model_fp32.xml',
    bin: 'test_model_fp32.bin',
    xmlURL:
      'https://raw.githubusercontent.com/openvinotoolkit/testdata/master/models/test_model/test_model_fp32.xml',
    binURL:
      'https://media.githubusercontent.com/media/openvinotoolkit/testdata/master/models/test_model/test_model_fp32.bin',
  },
};

module.exports = {
  getModelPath,
  downloadTestModel,
  isModelAvailable,
  testModels,
};

function getModelPath(isFP16 = false) {
  const modelName = `test_model_fp${isFP16 ? 16 : 32}`;

  return {
    xml: path.join(modelDir, `${modelName}.xml`),
    bin: path.join(modelDir, `${modelName}.bin`),
  };
}

async function downloadTestModel(model) {
  const modelsDir = './tests/unit/test_models';
  try {
    const ifModelsDirectoryExists = await checkIfPathExists(modelsDir);
    if (!ifModelsDirectoryExists) {
      await fs.mkdir(modelDir);
    }

    const modelPath = path.join(modelsDir, model.xml);
    const modelExists = await checkIfPathExists(modelPath);
    if (modelExists) return;

    const { env } = process;
    const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

    await downloadFile(model.xmlURL, modelsDir, model.xml, proxyUrl);
    await downloadFile(model.binURL, modelsDir, model.bin, proxyUrl);
  } catch(error) {
    console.error(`Failed to download the model: ${error}.`);
    throw error;
  }
}

async function isModelAvailable(model) {
  const baseArtifactsDir = './tests/unit/test_models';
  const modelPath = path.join(baseArtifactsDir, model.xml);
  const modelExists = await checkIfPathExists(modelPath);
  if (modelExists) return;

  console.log(
    '\n\nTestModel cannot be found.\nPlease run `npm run test_setup`.\n\n',
  );
  process.exit(1);
}
