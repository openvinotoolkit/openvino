// -*- coding: utf-8 -*-
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');
const fs = require('node:fs/promises');
const {
  downloadFile,
  checkIfPathExists,
} = require('../scripts/lib/utils');

const modelDir = 'tests/unit/test_models/';

function getModelPath(fileName) {
  return path.join(modelDir, fileName);
}

const testModels = {
  testModelFP32: {
    xml: getModelPath('test_model_fp32.xml'),
    bin: getModelPath('test_model_fp32.bin'),
    inputShape: [1, 3, 32, 32],
    outputShape: [1, 10],
    xmlURL:
      'https://raw.githubusercontent.com/openvinotoolkit/testdata/master/models/test_model/test_model_fp32.xml',
    binURL:
      'https://media.githubusercontent.com/media/openvinotoolkit/testdata/master/models/test_model/test_model_fp32.bin',
  },
  modelV3Small: {
    xml: getModelPath('v3-small_224_1.0_float.xml'),
    bin: getModelPath('v3-small_224_1.0_float.bin'),
    inputShape: [1, 224, 224, 3],
    outputShape: [1, 1001],
    xmlURL:
      'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.xml',
    binURL:
      'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.bin',
  },
};

module.exports = {
  compareModels,
  sleep,
  downloadTestModel,
  isModelAvailable,
  testModels,
  lengthFromShape,
};

function compareModels(model1, model2) {
  const differences = [];
  if (model1.getFriendlyName() !== model2.getFriendlyName()) {
    differences.push('Friendly names of models are not equal ' +
        `model_one: ${model1.getFriendlyName()},` +
        `model_two: ${model2.getFriendlyName()}`);
  }

  if (model1.inputs.length !== model2.inputs.length) {
    differences.push('Number of models\' inputs are not equal ' +
    `model_one: ${model1.inputs.length}, ` +
    `model_two: ${model2.inputs.length}`);
  }

  if (model1.outputs.length !== model2.outputs.length) {
    differences.push('Number of models\' outputs are not equal ' +
        `model_one: ${model1.outputs.length}, ` +
        `model_two: ${model2.outputs.length}`);
  }

  if (differences.length) {
    throw new Error(differences.join('\n'));
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function lengthFromShape(shape) {
  return shape.reduce(
    (accumulator, currentValue) => accumulator * currentValue,
    1
  );
}

async function downloadTestModel(model) {
  try {
    const ifModelsDirectoryExists = await checkIfPathExists(modelDir);
    if (!ifModelsDirectoryExists) {
      await fs.mkdir(modelDir);
    }

    const { env } = process;
    const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

    const modelExists = await checkIfPathExists(model.xml);
    if (!modelExists) await downloadFile(
      model.xmlURL,
      path.dirname(model.xml),
      path.basename(model.xml),
      proxyUrl,
    );

    const weightsExists = await checkIfPathExists(model.bin);
    if (!weightsExists) await downloadFile(
      model.binURL,
      path.dirname(model.bin),
      path.basename(model.bin),
      proxyUrl,
    );

  } catch(error) {
    console.error(`Failed to download the model: ${error}.`);
    throw error;
  }
}

async function isModelAvailable(model) {
  const modelExists = await checkIfPathExists(model.xml);
  if (modelExists) return;

  console.log(
    '\n\nTestModel cannot be found.\nPlease run `npm run test_setup`.\n\n',
  );
  process.exit(1);
}
