const os = require('os');
const path = require('path');
const fs = require('node:fs/promises');
const decompress = require('decompress');
const { createWriteStream } = require('node:fs');
const { HttpsProxyAgent } = require('https-proxy-agent');

const packageJson = require('../package.json');

const BINARY_PATH = process.env.BINARY_PATH || './bin';

const codeENOENT = 'ENOENT';

const relativeLibsDir = 'runtime/lib/intel64/';
const libsList = [
  'libopenvino_ir_frontend.so.2023.2.0',
  'libopenvino_ir_frontend.so.2320',
  'libopenvino.so.2320',
  'libopenvino.so.2023.2.0',
];

if (require.main === module) {
  main();
}

async function main() {
  const destinationPath = path.resolve(__dirname, '..', BINARY_PATH);
  const force = process.argv.includes('-f');
  const ignoreIfExists = process.argv.includes('--ignore-if-exists');
  const { env } = process;
  const proxy = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;
  try {
    await downloadRuntime(destinationPath, { force, ignoreIfExists, proxy });
  } catch (error) {
    if (error instanceof RuntimeExistsError) {
      console.error(
        `Directory '${destinationPath}' already exists. To force runtime downloading run 'npm run download_runtime -- -f'`
      );
    } else {
      throw error;
    }
    process.exit(1);
  }
}

class RuntimeExistsError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RuntimeExistsError';
    Error.captureStackTrace(this, RuntimeExistsError);
  }
}

/**
 * Download OpenVINO Runtime archive and extract it to destination directory.
 *
 * @async
 * @function downloadRuntime
 * @param {string} destinationPath - The destination directory path.
 * @param {Object} [config] - The configuration object.
 * @param {boolean} [config.force=false] - The flag to force install and replace runtime if it exists. Default is `false`.
 * @param {boolean} [config.ignoreIfExists=true] - The flag to skip installation if it exists Default is `true`.
 * @param {string|null} [config.proxy=null] - The proxy URL. Default is `null`.
 * @returns {Promise<void>}
 * @throws {RuntimeExistsError}
 */
async function downloadRuntime(destinationPath, config = { force: false, ignoreIfExists: true, proxy: null }) {
  const osName = getOsName();

  const isRuntimeDirectoryExists = await checkIfDirectoryExists(destinationPath);

  if (isRuntimeDirectoryExists && !config.force) {
    if (config.ignoreIfExists) {
      console.warn(
        `Directory '${destinationPath}' already exists. Skipping runtime downloading because 'ignoreIfExists' flag is passed.`
      );
      return;
    }

    throw new RuntimeExistsError(
      `Directory '${destinationPath}' already exists. To force runtime downloading use 'force' flag.`
    );
  }

  const runtimeArchiveUrl = getRuntimeArchiveUrl(osName, packageJson.binary[osName]);

  try {
    const filename = path.basename(runtimeArchiveUrl);
    const tempDirectoryPath = path.join(os.tmpdir(), 'temp-ov-runtime-archive');
    const archiveFilePath = path.resolve(tempDirectoryPath, filename);
    const tempOVRuntimePath = path.join(os.tmpdir(), 'temp-ov-runtime');

    await removeDirectory(tempDirectoryPath);
    await fs.mkdir(tempDirectoryPath);

    await removeDirectory(destinationPath);
    await fs.mkdir(destinationPath);

    console.log('Downloading OpenVINO runtime archive...');
    await downloadFile(runtimeArchiveUrl, filename, tempDirectoryPath, config.proxy);
    console.log('OpenVINO runtime archive downloaded.');

    await removeDirectory(tempOVRuntimePath);

    console.log('Extracting archive...');
    await decompress(archiveFilePath, tempOVRuntimePath, { strip: 1 });
    await removeDirectory(tempDirectoryPath);
    console.log('The archive was successfully extracted.');

    console.log('Сopying libraries:');
    for (const library of libsList) {
      const dest = path.join(destinationPath, library);

      await fs.copyFile(
        path.join(tempOVRuntimePath, relativeLibsDir, library), 
        dest,
      );
      console.log(`Copied: ${dest}`);
    }
    console.log('Copying completed.');
  } catch (error) {
    console.error(`Failed to download OpenVINO runtime: ${error}.`);
    throw error;
  }
}

/**
 * The OS information object.
 * @typedef {Object} OsInfo
 * @property {NodeJS.Platform} platform
 * @property {string} os
 * @property {string} dir
 * @property {string} letter
 * @property {string} extension
 */

/**
 * Get OS name.
 *
 * @async
 * @function getOsName
 * @returns {string}
 */
function getOsName() {
  const platform = os.platform();

  if (!['win32', 'linux', 'darwin'].includes(platform)) {
    throw new Error(`Platform '${platform}' is not supported.`);
  }

  const osArchiveMapping = {
    win32: 'windows',
    linux: 'linux',
    darwin: 'macos',
  };

  const arch = os.arch();

  if (!['arm64', 'armhf', 'x64'].includes(arch)) {
    throw new Error(`Architecture '${arch}' is not supported.`);
  }

  const detectedOs = osArchiveMapping[platform];

  if (!detectedOs) {
    throw new Error('Failed to detect OS.');
  }

  return detectedOs;
}

/**
 * Check if directory exists.
 *
 * @async
 * @function checkIfDirectoryExists
 * @param {string} directoryPath - The directory path.
 * @returns {Promise<boolean>}
 */
async function checkIfDirectoryExists(directoryPath) {
  try {
    await fs.access(directoryPath);
    return true;
  } catch (error) {
    if (error.code === codeENOENT) {
      return false;
    }
    throw error;
  }
}

/**
 * Get full URL to OpenVINO runtime archive.
 *
 * @function getRuntimeArchiveUrl
 * @param {string} os - OS name
 * @param {string} archiveName - name of related archive
 * @returns {string}
 */
function getRuntimeArchiveUrl(dir, archiveName) {
  const url = packageJson.binary['url'];
  const version = packageJson.binary['version'];

  return new URL(path.join(url, version, dir, archiveName)).toString();
}

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
    await fs.rm(path, { recursive: true, force: true });
  } catch (error) {
    if (error.code !== codeENOENT) {
      throw error;
    }
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
 * @returns {Promise<void>}
 */
function downloadFile(url, filename, destination, proxy = null) {
  const timeout = 5000;
  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);
  const httpModule = new URL(url).protocol === 'https:' ? require('node:https') : require('node:http');

  let agent;

  if (proxy) {
    agent = new HttpsProxyAgent(proxy);
    console.log(`Proxy agent is configured with '${proxy}'.`);
  }

  return new Promise((resolve, reject) => {
    file.on('error', (error) => {
      reject(`Failed to open file stream: ${error}.`);
    });

    const request = httpModule.get(url, { agent }, (res) => {
      const { statusCode } = res;

      if (statusCode !== 200) {
        return reject(`Server returned status code ${statusCode}.`);
      }

      res.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`File was successfully downloaded to '${fullPath}'.`);
        resolve();
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

module.exports = { downloadRuntime };
