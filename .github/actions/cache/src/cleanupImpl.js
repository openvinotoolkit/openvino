const core = require('@actions/core')
const fs = require('fs')
const path = require('path')
const {
  getSortedCacheFiles,
  humanReadableFileSize,
  calculateTotalSize
} = require('./utils')

// Function to remove old files if their combined size exceeds 50 GB
async function cleanUp() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true })
    const key = core.getInput('key', { required: true })
    const keysRestore = core
      .getInput('restore-keys', { required: false })
      .split('\n')
      .map(s => s.replace(/^!\s+/, '!').trim())
      .filter(x => x !== '')
    const maxCacheSize = core.getInput('max-cache-size', { required: false })

    // Minimum time peroid in milliseconds when the files was not used
    const minAccessTime = 1 * 60 * 60 * 1000 // 1 hour
    const currentDate = new Date()
    const minAccessDateAgo = new Date(currentDate - minAccessTime)

    core.debug(`cache-path: ${cacheRemotePath}`)
    core.debug(`key: ${key}`)
    core.debug(`restore-keys: ${keysRestore}`)

    var keyPattern = key
    if (keysRestore && keysRestore.length) {
      keyPattern = keysRestore.join('|')
    }

    const files = await getSortedCacheFiles(cacheRemotePath, keyPattern)
    let totalSize = await calculateTotalSize(cacheRemotePath, files)
    let maxCacheSizeInBytes = maxCacheSize * 1024 * 1024 * 1024

    if (totalSize > maxCacheSizeInBytes) {
      core.info(
        `The cache storage size ${humanReadableFileSize(totalSize)} exceeds allowed size ${humanReadableFileSize(maxCacheSizeInBytes)}`
      )
      for (let i = files.length - 1; i >= 0; i--) {
        var file = files[i]
        const filePath = path.join(cacheRemotePath, file)
        const fileStats = fs.statSync(filePath)

        if (fileStats.isFile() && fileStats.atime < minAccessDateAgo) {
          core.info(`Removing file: ${filePath}`)
          // fs.unlink(filePath)
          fs.unlink(filePath, err => {
            if (err) {
              core.warning(`Could not remove file: ${filePath}: ${err}`)
            } else {
              core.info(`{filePath} removed successfully`)
              totalSize -= fileStats.size
            }
          })
          // totalSize -= fileStats.size
        }

        if (totalSize <= maxCacheSizeInBytes) {
          // Check if total size
          break // Exit loop if total size is within limit
        }
      }
      core.info('Old cache files removed successfully')
    } else {
      core.info(
        `The cache storage size ${humanReadableFileSize(totalSize)} less then allowed size ${humanReadableFileSize(maxCacheSizeInBytes)}`
      )
    }
  } catch (error) {
    core.setFailed('Error removing old cache files.' + error.message)
  }
}

module.exports = {
  cleanUp
}
