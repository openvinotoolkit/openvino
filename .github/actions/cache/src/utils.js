const core = require('@actions/core')
const fs = require('fs')

async function getSortedCacheFiles(path, key = '') {
  if (!fs.existsSync(path)) {
    core.warning(`${path} doesn't exist`)
    return []
  }

  const cache_pattern = new RegExp(`^((${key}).*[.]cache)$`)

  const files = await fs.promises.readdir(path)
  filesSorded = files
    .filter(fileName => cache_pattern.test(fileName))
    .map(fileName => ({
      name: fileName,
      time: fs.statSync(`${path}/${fileName}`).mtime.getTime()
    }))
    .sort((a, b) => b.time - a.time)
    .map(file => file.name)

  core.debug(
    filesSorded.map(fileName => ({
      name: fileName,
      time: fs.statSync(`${path}/${fileName}`).mtime.getTime()
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

  return sizeInBytes.toFixed(2) + ' ' + units[id]
}

// Function to calculate the total size of files in bytes
async function calculateTotalSize(dir, files) {
  let totalSize = 0

  for (const file of files) {
    const filePath = path.join(dir, file)
    const fileStats = await stat(filePath)

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
