const core = require('@actions/core')
const fs = require('fs')
const path = require('path')

async function getSortedCacheFiles(cachePath, key = '') {
  if (!fs.existsSync(cachePath)) {
    core.warning(`${cachePath} doesn't exist`)
    return []
  }

  const cachePattern = new RegExp(`^((${key}).*[.]cache)$`)

  const files = await fs.promises.readdir(cachePath)
  const filesSorded = files
    .filter(fileName => cachePattern.test(fileName))
    .map(fileName => ({
      name: fileName,
      time: fs.statSync(path.join(cachePath, fileName)).atime.getTime()
    }))
    .sort((a, b) => b.time - a.time)
    .map(file => file.name)

  core.debug(
    filesSorded.map(fileName => ({
      name: fileName,
      time: fs.statSync(path.join(cachePath, fileName)).atime.getTime()
    }))
  )
  return filesSorded
}

function humanReadableFileSize(sizeInBytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let id = 0

  while (sizeInBytes >= 1024 && id < units.length - 1) {
    sizeInBytes /= 1024
    id++
  }

  return `${sizeInBytes.toFixed(2)} ${units[id]}`
}

// Function to calculate the total size of files in bytes
async function calculateTotalSize(dir, files) {
  let totalSize = 0

  for (const file of files) {
    const filePath = path.join(dir, file)
    const fileStats = fs.statSync(filePath)

    if (fileStats.isFile()) {
      totalSize += fileStats.size
    }
  }
  return totalSize
}

module.exports = {
  getSortedCacheFiles,
  humanReadableFileSize,
  calculateTotalSize
}
