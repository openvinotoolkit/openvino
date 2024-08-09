let https = require('follow-redirects').https;
const fs = require('fs');
const path = require('path');
const { HttpsProxyAgent } = require('https-proxy-agent');

async function main() {
  const baseArtifactsDir = './tests/unit/test_models';
  const modelName = 'test_model_fp32';
  const modelXMLName = `${modelName}.xml`;
  const modelBINName = `${modelName}.bin`;
  const baseURL = 'https://github.com/openvinotoolkit/testdata/raw/master/models/test_model/';

  const { env } = process;
  const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;
  let agent;
  if (proxyUrl) {
    agent = new HttpsProxyAgent(proxyUrl);
    console.log(`Proxy agent configured using: '${proxyUrl}'`);
  }

  downloadFile(baseURL + modelXMLName, path.join(baseArtifactsDir, modelXMLName), agent);
  downloadFile(baseURL + modelBINName, path.join(baseArtifactsDir, modelBINName), agent);
}

const downloadFile = (url, destination, agent) => {
  const file = fs.createWriteStream(destination);

  const options = {
    agent: agent,
  };

  https.get(url, options, (response) => {
    if (response.statusCode !== 200) {
      console.error(`Failed to get '${url}' (${response.statusCode})`);
      response.resume();
      return;
    }

    response.pipe(file);

    file.on('finish', () => {
      file.close(() => {
        console.log('Download completed!');
      });
    });
  }).on('error', (err) => {
    fs.unlink(destination, () => {});
    console.error(`Error downloading file: ${err.message}`);
  });
};

if (require.main === module) {
  main();
}
