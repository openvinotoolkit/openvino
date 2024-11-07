const os = require('node:os');
const tar = require('tar-fs');
const path = require('node:path');
const https = require('node:https');
const gunzip = require('gunzip-maybe');
const fs = require('node:fs/promises');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { createReadStream, createWriteStream } = require('node:fs');

const packageJson = require('../package.json');

const codeENOENT = 'ENOENT';

class BinaryManager {
  constructor(version, binaryConfig) {
    this.version = version;
    this.binaryConfig = binaryConfig;
  }

  getPlatformLabel() {
    return os.platform();
  }

  getArchLabel() {
    return os.arch();
  }

  getExtension() {
    return 'tar.gz';
  }

  getArchiveUrl() {
    const {
      host,
      package_name: packageNameTemplate,
      remote_path: remotePathTemplate,
    } = this.binaryConfig;
    const fullPathTemplate = `${remotePathTemplate}${packageNameTemplate}`
    const fullPath = fullPathTemplate
      .replace(new RegExp('{version}', 'g'), this.version)
      .replace(new RegExp('{platform}', 'g'), this.getPlatformLabel())
      .replace(new RegExp('{arch}', 'g'), this.getArchLabel())
      .replace(new RegExp('{extension}', 'g'), this.getExtension());

    return new URL(fullPath, host).toString();
  }

  getDestinationPath() {
    const modulePath = this.binaryConfig['module_path'];

    return path.resolve(__dirname, '..', modulePath);
  }

  static async prepareBinary(version, binaryConfig, options) {
    const binaryManager = new this(version, binaryConfig);
    const destinationPath = binaryManager.getDestinationPath();
    const isRuntimeDirectoryExists = await checkIfPathExists(destinationPath);

    if (isRuntimeDirectoryExists && !options.force) {
      if (options.ignoreIfExists) {
        console.warn(
          `Directory '${destinationPath}' already exists. Skipping `
          + 'runtime downloading because "ignoreIfExists" flag is passed.'
        );

        return;
      }

      throw new Error(
        `Directory '${destinationPath}' already exists. ` +
          'To force runtime downloading use "force" flag.',
      );
    }

    const archiveUrl = binaryManager.getArchiveUrl();
    let tempDirectoryPath = null;

    try {
      tempDirectoryPath = await fs.mkdtemp(
        path.join(os.tmpdir(), 'temp-ov-runtime-archive-')
      );

      const filename = path.basename(archiveUrl);

      console.log('Downloading OpenVINO runtime archive...');
      const archiveFilePath = await downloadFile(
        archiveUrl,
        tempDirectoryPath,
        filename,
        options.proxy,
      )
      console.log('OpenVINO runtime archive downloaded.');

      await this.unarchive(archiveFilePath, destinationPath);
      console.log('The archive was successfully extracted.');
    } catch(error) {
      console.error(`Failed to download OpenVINO runtime: ${error}.`);
      throw error;
    } finally {
      if (tempDirectoryPath) await removeDirectory(tempDirectoryPath);
    }
  }

  static isCompatible() {
    const missleadings = [];
    const platform = os.platform();

    if (!['win32', 'linux', 'darwin'].includes(platform))
      missleadings.push(`Platform '${platform}' is not supported.`);

    const arch = os.arch();

    if (!['arm64', 'armhf', 'x64'].includes(arch))
      missleadings.push(`Architecture '${arch}' is not supported.`);

    if (platform === 'win32' && arch !== 'x64')
      missleadings.push(`Version for windows and '${arch}' is not supported.`);

    if (missleadings.length) {
      console.error(missleadings.join(' '));
      return false;
    }

    return true;
  }

  /**
   * Unarchive tar and tar.gz archives.
   *
   * @function unarchive
   * @param {string} archivePath - Path to archive.
   * @param {string} dest - Path where to unpack.
   * @returns {Promise<void>}
   */
  static unarchive(archivePath, dest) {
    return new Promise((resolve, reject) => {
      createReadStream(archivePath)
        .pipe(gunzip())
        .pipe(tar.extract(dest)
          .on('finish', () => {
            resolve();
          }).on('error', (err) => {
            reject(err);
          }),
        );
    });
  }
}

if (require.main === module) main();

module.exports = BinaryManager;

async function main() {
  if (!BinaryManager.isCompatible()) process.exit(1);

  const force = process.argv.includes('-f');
  const ignoreIfExists = process.argv.includes('--ignore-if-exists');

  const { env } = process;
  const proxy = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  await BinaryManager.prepareBinary(
    packageJson.version,
    packageJson.binary,
    { force, ignoreIfExists, proxy },
  );
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
  } catch (error) {
    if (error.code === codeENOENT) console.log(`Path: ${path} doesn't exist`);

    throw error;
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
