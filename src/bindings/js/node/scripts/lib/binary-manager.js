const os = require('node:os');
const tar = require('tar-fs');
const path = require('node:path');
const gunzip = require('gunzip-maybe');
const fs = require('node:fs/promises');
const { createReadStream } = require('node:fs');

const { downloadFile, checkIfPathExists, removeDirectory } = require('./utils');

class BinaryManager {
  constructor(packageRoot, version, binaryConfig) {
    this.packageRoot = packageRoot;
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

    return path.resolve(this.packageRoot, modulePath);
  }

  /**
   * Prepares the binary by downloading and extracting the OpenVINO runtime archive.
   *
   * @param {string} packageRoot - The root directory of the package.
   * @param {string} version - The version of the binary.
   * @param {Object} binaryConfig - The configuration object for the binary.
   * @param {Object} options - The options for preparing the binary.
   * @param {boolean} options.force - Whether to force the download if the directory already exists.
   * @param {boolean} options.ignoreIfExists - Whether to ignore the download if the directory already exists.
   * @param {string} [options.proxy] - The proxy to use for downloading the file.
   * @throws {Error} If the directory already exists and the force option is not set.
   * @throws {Error} If the download or extraction fails.
   * @returns {Promise<void>} A promise that resolves when the binary is prepared.
   */
  static async prepareBinary(packageRoot, version, binaryConfig, options) {
    const binaryManager = new this(packageRoot, version, binaryConfig);
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

      await removeDirectory(destinationPath);
      await this.unarchive(archiveFilePath, destinationPath);
      console.log('The archive was successfully extracted.');
    } catch(error) {
      console.error(`Failed to download OpenVINO runtime: ${error}.`);
      throw error;
    } finally {
      if (tempDirectoryPath) await removeDirectory(tempDirectoryPath);
    }
  }

  /**
   * Checks if the current platform and architecture are compatible.
   *
   * Supported platforms: 'win32', 'linux', 'darwin'.
   * Supported architectures: 'arm64', 'armhf', 'x64'.
   *
   * If the platform or architecture is not supported, an error message is logged to the console.
   *
   * @returns {boolean} Returns true if the platform and architecture are compatible, otherwise false.
   */
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

module.exports = BinaryManager;
