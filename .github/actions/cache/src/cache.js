const { log } = require('console')
const fs = require('fs')

const cache_pattern = new RegExp('^(.*[.]cache)$')

async function getSortedCacheFiles(path) {
  log(`!!!Path: ${path}!!!`)
  const files = await fs.promises.readdir(path)

  log(files)
  filesSorded = files
    .filter(fileName => cache_pattern.test(fileName))
    .map(fileName => ({
      name: fileName,
      time: fs.statSync(`${path}/${fileName}`).mtime.getTime()
    }))
    .sort((a, b) => b.time - a.time)
    .map(file => file.name)
  log(
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
