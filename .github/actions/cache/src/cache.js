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

module.exports = {
  getSortedCacheFiles
}
