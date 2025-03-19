const core = require('@actions/core');
const fs = require('fs');
const path = require('path');

async function getSortedCacheFiles(cachePath, key = '', recursive = false) {
  if (!(await checkFileExists(cachePath))) {
    core.warning(`${cachePath} doesn't exist`);
    return [];
  }

  const cachePattern = new RegExp(`^((${key}).*[.]cache)$`);

  const filesSorted = (await fs.promises.readdir(cachePath, { recursive }))
    .filter(fileName => cachePattern.test(path.basename(fileName)))
    .map(fileName => ({
      name: fileName,
      time: fs.statSync(path.join(cachePath, fileName)).atimeMs
    }))
    .sort((a, b) => b.time - a.time)
    .map(file => file.name);

  core.debug(
    filesSorted.map(fileName => ({
      name: fileName,
      atime: fs.statSync(path.join(cachePath, fileName)).atimeMs
    }))
  );
  return filesSorted;
}

function humanReadableFileSize(sizeInBytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let id = 0;

  while (sizeInBytes >= 1024 && id < units.length - 1) {
    sizeInBytes /= 1024;
    id++;
  }

  return `${sizeInBytes.toFixed(2)} ${units[id]}`;
}

// Function to calculate the total size of files in bytes
async function calculateTotalSize(dir, files) {
  let totalSize = 0;

  for (const file of files) {
    const filePath = path.join(dir, file);
    const fileStats = await fs.promises.stat(filePath);

    if (fileStats.isFile()) {
      totalSize += fileStats.size;
    }
  }
  return totalSize;
}

async function checkFileExists(filePath) {
  try {
    await fs.promises.access(filePath);
    return true;
  } catch (error) {
    return false;
  }
}

module.exports = {
  getSortedCacheFiles,
  humanReadableFileSize,
  calculateTotalSize,
  checkFileExists
};
