const os = require('os');
const path = require('path');
const fs = require('node:fs/promises');
const decompress = require('decompress');
const { createWriteStream } = require('node:fs');
const { HttpsProxyAgent } = require('https-proxy-agent');

const packageJson = require('../package.json');

const codeENOENT = 'ENOENT';

if (require.main === module) {
  main();
}

async function main() {
  const modulePath = packageJson.binary['module_path'];
  const destinationPath = path.resolve(__dirname, '..', modulePath);
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
  const osInfo = await getOsInfo();

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

  const runtimeArchiveUrl = getRuntimeArchiveUrl(osInfo);

  try {
    const filename = path.basename(runtimeArchiveUrl);
    const tempDirectoryPath = path.join(os.tmpdir(), 'temp-ov-runtime-archive');
    const archiveFilePath = path.resolve(tempDirectoryPath, filename);

    await removeDirectory(tempDirectoryPath);
    await fs.mkdir(tempDirectoryPath);

    console.log('Downloading OpenVINO runtime archive...');
    await downloadFile(runtimeArchiveUrl, filename, tempDirectoryPath, config.proxy);
    console.log('OpenVINO runtime archive downloaded.');

    await removeDirectory(destinationPath);

    console.log('Extracting archive...');
    await decompress(archiveFilePath, destinationPath, { strip: 1 });
    await removeDirectory(tempDirectoryPath);
    console.log('The archive was successfully extracted.');
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
 * Get information about OS.
 *
 * @async
 * @function getOsInfo
 * @returns {Promise<OsInfo>}
 */
async function getOsInfo() {
  const platform = os.platform();

  if (!['win32', 'linux', 'darwin'].includes(platform)) {
    throw new Error(`Platform '${platform}' is not supported.`);
  }

  const osArchiveMapping = {
    win32: {
      dir: 'windows',
      letter: 'w',
      extension: 'zip',
    },
    linux: {
      dir: 'linux',
      letter: 'l',
      extension: 'tgz',
    },
    darwin: {
      dir: 'macos',
      letter: 'm',
      extension: 'tgz',
    },
  };

  const arch = os.arch();

  if (!['arm64', 'armhf', 'x64'].includes(arch)) {
    throw new Error(`Architecture '${arch}' is not supported.`);
  }

  const archMapping = {
    arm64: 'arm64',
    armhf: 'armhf',
    x64: 'x86_64',
  };

  let detectedOs = null;

  switch (platform) {
    case 'linux':
      const osReleaseData = await fs.readFile('/etc/os-release', 'utf8');

      detectedOs = osReleaseData.includes('Ubuntu 22')
        ? 'ubuntu22'
        : osReleaseData.includes('Ubuntu 20')
        ? 'ubuntu20'
        : osReleaseData.includes('Ubuntu 18')
        ? 'ubuntu18'
        : ['arm64', 'armhf'].includes(arch) && osReleaseData.includes('ID=debian')
        ? 'debian9'
        : null;

      break;

    case 'darwin':
      const [major, minor] = os.release().split('.');

      detectedOs = major === 10 && minor >= 15 ? 'macos_10_15' : major === 11 ? 'macos_11_0' : null;

      break;

    case 'win32':
      detectedOs = 'windows';

      break;
  }

  if (!detectedOs) {
    throw new Error('Failed to detect OS.');
  }

  return {
    platform,
    os: detectedOs,
    arch: archMapping[arch],
    ...osArchiveMapping[platform],
  };
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
 * Get OpenVINO runtime archive URL.
 *
 * @function getRuntimeArchiveUrl
 * @param {OsInfo} osInfo - The OS related data.
 * @returns {string}
 */
function getRuntimeArchiveUrl(osInfo) {
  const packageNameTemplate = packageJson.binary['package_name'];

  const packageName = packageNameTemplate
    .replace('{letter}', osInfo.letter)
    .replace('{os}', osInfo.os)
    .replace('{extension}', osInfo.extension)
    .replace('{arch}', osInfo.arch)
    .replace('{version}', packageJson.binary.version);

  const { host, remote_path } = packageJson.binary;

  return new URL(path.join(remote_path, osInfo.dir, packageName), host).toString();
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
