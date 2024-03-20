const core = require('@actions/core')
const fs = require('fs')

async function getSortedCacheFiles(path, key = '') {
  if (!fs.existsSync(path)) {
    core.warning(`${path} doesn't exist`)
    return []
  }

  const cache_pattern = new RegExp(`^(${key}.*[.]cache)$`)

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
      time: fs.statSync(`${path}/${fileName}`).atime.getTime()
    }))
  )
  return filesSorded
}

function humanReadableFileSize(sizeInBytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let index = 0

  while (sizeInBytes >= 1024 && index < units.length - 1) {
    sizeInBytes /= 1024
    index++
  }

  return sizeInBytes.toFixed(2) + ' ' + units[index]
}

module.exports = {
  getSortedCacheFiles,
  humanReadableFileSize
}
