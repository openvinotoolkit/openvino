// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const path = require('path');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { createWriteStream } = require('node:fs');
const { mkdir, stat } = require('node:fs/promises');

module.exports = {
  getModelPath,
  downloadFile,
};

function getModelPath(isFP16=false) {
  const basePath = 'tests/unit/test_models/';
  const modelName = `test_model_fp${isFP16 ? 16 : 32}`;

  return {
    xml: path.join(basePath, `${modelName}.xml`),
    bin: path.join(basePath, `${modelName}.bin`),
  };
}

async function downloadFile(url, filename, destination) {
  const { env } = process;
  const timeout = 5000;

  await applyFolderPath(destination);

  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);
  const protocolString = new URL(url).protocol === 'https:' ? 'https' : 'http';
  const module = require(`node:${protocolString}`);
  const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  let agent;

  if (proxyUrl) {
    agent = new HttpsProxyAgent(proxyUrl);
    console.log(`Proxy agent configured using: '${proxyUrl}'`);
  }

  return new Promise((resolve, reject) => {
    file.on('error', e => {
      reject(`Error oppening file stream: ${e}`);
    });

    const getRequest = module.get(url, { agent }, res => {
      const { statusCode } = res;

      if (statusCode !== 200)
        return reject(`Server returns status code: ${statusCode}`);

      res.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`File successfully stored at '${fullPath}'`);
        resolve();
      });
    });

    getRequest.on('error', e => {
      reject(`Error sending request: ${e}`);
    });

    getRequest.setTimeout(timeout, () => {
      getRequest.destroy();
      reject(`Request timed out after ${timeout}`);
    });
  });
}

async function applyFolderPath(dirPath) {
  try {
    await stat(dirPath);

    return;
  } catch(err) {
    if (err.code !== 'ENOENT') throw err;

    await mkdir(dirPath, { recursive: true });
  }
}
