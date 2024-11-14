const path = require('node:path');
const https = require('node:https');
const fs = require('node:fs/promises');
const { createWriteStream } = require('node:fs');

const { HttpsProxyAgent } = require('https-proxy-agent');

const codeENOENT = 'ENOENT';

module.exports = {
  removeDirectory,
  checkIfPathExists,
  downloadFile,
};

/**
 * Remove directory and its content.
 *
 * @async
 * @function removeDirectory
 * @param {string} path - The directory path.
 * @returns {Promise<void>}
 */
async function removeDirectory(path) {
  try {
    console.log(`Removing ${path}`);
    await fs.rm(path, { recursive: true });
  } catch (error) {
    if (error.code !== codeENOENT) throw error;

    console.warn(`Path: ${path} doesn't exist`);
  }
}

/**
 * Check if path exists.
 *
 * @async
 * @function checkIfPathExists
 * @param {string} path - The path to directory or file.
 * @returns {Promise<boolean>}
 */
async function checkIfPathExists(path) {
  try {
    await fs.access(path);
    return true;
  } catch (error) {
    if (error.code === codeENOENT) {
      return false;
    }
    throw error;
  }
}

/**
 * Download file by URL and save it to the destination path.
 *
 * @function downloadFile
 * @param {string} url - The file URL.
 * @param {string} filename - The filename of result file.
 * @param {string} destination - The destination path of result file.
 * @param {string} [proxy=null] - (Optional) The proxy URL.
 * @returns {Promise<string>} - Path to downloaded file.
 */
function downloadFile(url, destination, filename, proxy = null) {
  console.log(`Downloading file by link: ${url} to ${destination}`
    + `with filename: ${filename}`);

  const timeout = 5000;
  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);

  if (new URL(url).protocol === 'http:')
    throw new Error('Http link doesn\'t support');

  let agent;

  if (proxy) {
    agent = new HttpsProxyAgent(proxy);
    console.log(`Proxy agent is configured with '${proxy}'.`);
  }

  return new Promise((resolve, reject) => {
    file.on('error', (error) => {
      reject(`Failed to open file stream: ${error}.`);
    });

    console.log(`Download file by link: ${url}`);

    const request = https.get(url, { agent }, (res) => {
      const { statusCode } = res;

      if (statusCode !== 200) {
        return reject(`Server returned status code ${statusCode}.`);
      }

      res.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`File was successfully downloaded to '${fullPath}'.`);
        resolve(fullPath);
      });
    });

    request.on('error', (error) => {
      reject(`Failed to send request: ${error}.`);
    });

    request.setTimeout(timeout, () => {
      request.destroy();
      reject(`Request was timed out after ${timeout} ms.`);
    });
  });
}
