const core = require('@actions/core');
const fs = require('fs/promises');
const path = require('path');
const {
  getSortedCacheFiles,
  humanReadableFileSize,
  calculateTotalSize
} = require('./utils');

// Function to remove old files if their combined size exceeds the allowed size
async function cleanUp() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true });
    const key = core.getInput('key', { required: true });
    const keysRestore = core
      .getInput('restore-keys', { required: false })
      .split('\n')
      .map(s => s.replace(/^!\s+/, '!').trim())
      .filter(x => x !== '');
    const cacheSize = core.getInput('cache-size', { required: false });
    const cacheMaxSize = core.getInput('max-cache-size', { required: false });
    const recursive = core.getInput('recursive', { required: false });

    // Minimum time peroid in milliseconds when the files was not useds
    const minAccessTime = 7 * 24 * 60 * 60 * 1000; // 1 week
    const currentDate = new Date();
    const minAccessDateAgo = new Date(currentDate - minAccessTime);

    core.debug(`cache-path: ${cacheRemotePath}`);
    core.debug(`key: ${key}`);
    core.debug(`restore-keys: ${keysRestore}`);
    core.debug(`cache-size: ${cacheSize}`);
    core.debug(`max-cache-size: ${cacheMaxSize}`);
    core.debug(`recursive: ${recursive}`);

    let keyPattern = key;
    if (keysRestore && keysRestore.length) {
      keyPattern = keysRestore.join('|');
    }

    const files = await getSortedCacheFiles(
      cacheRemotePath,
      keyPattern,
      recursive
    );
    const minCacheSizeInBytes = cacheSize * 1024 * 1024 * 1024;
    const maxCacheSizeInBytes = cacheMaxSize * 1024 * 1024 * 1024;
    let totalSize = await calculateTotalSize(cacheRemotePath, files);

    if (totalSize <= minCacheSizeInBytes) {
      core.info(
        `The cache storage size ${humanReadableFileSize(totalSize)} less then the allowed size ${humanReadableFileSize(minCacheSizeInBytes)}`
      );
      return;
    }

    core.info(
      `The cache storage size ${humanReadableFileSize(totalSize)} exceeds allowed size ${humanReadableFileSize(minCacheSizeInBytes)}`
    );
    for (const file of files.reverse()) {
      const filePath = path.join(cacheRemotePath, file);
      const fileStats = await fs.stat(filePath);

      // skipping recently used files if total cache size less then maxCacheSizeInBytes
      if (
        !fileStats.isFile() ||
        (fileStats.atime >= minAccessDateAgo && totalSize < maxCacheSizeInBytes)
      ) {
        core.info(`Skipping: ${filePath}`);
        continue;
      }

      core.info(`Removing file: ${filePath}`);
      await fs.unlink(filePath);
      core.info(`${filePath} removed successfully`);
      totalSize -= fileStats.size;

      // Exit if total size is within limit
      if (totalSize <= minCacheSizeInBytes) {
        core.info('Old cache files removed successfully');
        return;
      }
    }
  } catch (error) {
    core.error(`Error removing old cache files: ${error.message}`);
  }
}

module.exports = {
  cleanUp
};
