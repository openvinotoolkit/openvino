const os = require('os');
const path = require('path');
const tar = require('tar-fs');
const https = require('node:https');
const gunzip = require('gunzip-maybe');
const fs = require('node:fs/promises');
const { createReadStream, createWriteStream } = require('node:fs');
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
  } catch(error) {
    if (error instanceof RuntimeExistsError) {
      console.error(
        `Directory '${destinationPath}' already exists. ` +
          'To force runtime downloading run \'npm run download_runtime -- -f\'',
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
 * @param {boolean} [config.force=false] - The flag
 * to force install and replace runtime if it exists. Default is `false`.
 * @param {boolean} [config.ignoreIfExists=true] - The flag
 * to skip installation if it exists Default is `true`.
 * @param {string|null} [config.proxy=null] - The proxy URL. Default is `null`.
 * @returns {Promise<void>}
 * @throws {RuntimeExistsError}
 */
async function downloadRuntime(
  destinationPath,
  config = { force: false, ignoreIfExists: true, proxy: null },
) {
  const { version } = packageJson;
  const osInfo = await getOsInfo();
  const isRuntimeDirectoryExists = await checkIfPathExists(destinationPath);

  if (isRuntimeDirectoryExists && !config.force) {
    if (config.ignoreIfExists) {
      console.warn(
        `Directory '${destinationPath}' already exists. Skipping ` +
          'runtime downloading because \'ignoreIfExists\' flag is passed.',
      );

      return;
    }

    throw new RuntimeExistsError(
      `Directory '${destinationPath}' already exists. ` +
        'To force runtime downloading use \'force\' flag.',
    );
  }

  const runtimeArchiveUrl = getRuntimeArchiveUrl(version, osInfo);
  const tmpDir = `temp-ov-runtime-archive-${new Date().getTime()}`;
  const tempDirectoryPath = path.join(os.tmpdir(), tmpDir);

  try {
    const filename = path.basename(runtimeArchiveUrl);
    const archiveFilePath = path.resolve(tempDirectoryPath, filename);

    await fs.mkdir(tempDirectoryPath);

    console.log('Downloading OpenVINO runtime archive...');
    await downloadFile(
      runtimeArchiveUrl,
      tempDirectoryPath,
      filename,
      config.proxy,
    );
    console.log('OpenVINO runtime archive downloaded.');

    await removeDirectory(destinationPath);

    console.log('Extracting archive...');
    await unarchive(archiveFilePath, destinationPath);

    console.log('The archive was successfully extracted.');
  } catch(error) {
    console.error(`Failed to download OpenVINO runtime: ${error}.`);
    throw error;
  } finally {
    await removeDirectory(tempDirectoryPath);
  }
}

/**
 * The OS information object.
 * @typedef {Object} OsInfo
 * @property {NodeJS.Platform} platform
 * @property {string} arch
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

  const arch = os.arch();

  if (!['arm64', 'armhf', 'x64'].includes(arch)) {
    throw new Error(`Architecture '${arch}' is not supported.`);
  }

  if (platform === 'win32' && arch !== 'x64') {
    throw new Error(`Version for windows and '${arch}' is not supported.`);
  }

  return { platform, arch };
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
  } catch(error) {
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
 * @param {string} version - Package version.
 * @param {OsInfo} osInfo - The OS related data.
 * @returns {string}
 */
function getRuntimeArchiveUrl(version, osInfo) {
  const {
    host,
    package_name: packageNameTemplate,
    remote_path: remotePathTemplate,
  } = packageJson.binary;
  const fullPathTemplate = `${remotePathTemplate}${packageNameTemplate}`;
  const fullPath = fullPathTemplate
    .replace(new RegExp('{version}', 'g'), version)
    .replace(new RegExp('{platform}', 'g'), osInfo.platform)
    .replace(new RegExp('{arch}', 'g'), osInfo.arch);

  return new URL(fullPath, host).toString();
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
    console.log(`Removing ${path}`);
    await fs.rm(path, { recursive: true, force: true });
  } catch(error) {
    if (error.code === codeENOENT) console.log(`Path: ${path} doesn't exist`);

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
 * @returns {Promise<void>}
 */
function downloadFile(url, destination, filename, proxy = null) {
  const timeout = 5000;
  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);

  if (new URL(url).protocol === 'http')
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

/**
 * Unarchive tar and tar.gz archives.
 *
 * @function unarchive
 * @param {tarFilePath} tarFilePath - Path to archive.
 * @param {dest} tarFilePath - Path where to unpack.
 * @returns {Promise<void>}
 */
function unarchive(tarFilePath, dest) {
  return new Promise((resolve, reject) => {
    createReadStream(tarFilePath)
      .pipe(gunzip())
      .pipe(
        tar
          .extract(dest)
          .on('finish', () => {
            resolve();
          })
          .on('error', (err) => {
            reject(err);
          }),
      );
  });
}

module.exports = { downloadRuntime, downloadFile, checkIfPathExists };
