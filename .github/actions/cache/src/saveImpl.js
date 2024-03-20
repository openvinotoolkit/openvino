const core = require('@actions/core')
const tar = require('tar')
const fs = require('fs')
const path = require('path')
const {
  getSortedCacheFiles,
  humanReadableFileSize,
  calculateTotalSize
} = require('./utils')
/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function save() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true })
    const toCachePath = core.getInput('path', { required: true })
    const key = core.getInput('key', { required: true })

    core.debug(`cache-path: ${cacheRemotePath}`)
    core.debug(`path: ${toCachePath}`)
    core.debug(`key: ${key}`)

    if (!key) {
      core.warning(`Key ${key} is not specified.`)
      return
    }

    var tarName = `${key}.cache`
    var tarPath = path.join(cacheRemotePath, tarName)

    if (fs.existsSync(tarPath)) {
      core.warning(`Cache file ${tarName} already exists`)
      return
    }

    core.info(`Preparing cache archive ${tarName}`)
    tar.c(
      {
        gzip: true,
        file: tarName,
        cwd: toCachePath,
        sync: true
      },
      ['.']
    )
    core.info('Copying cache...')
    fs.copyFileSync(tarName, tarPath)
    core.info(`${tarName} was copied to ${tarPath}`)

    core.setOutput('cache-file', tarName)
    core.setOutput('cache-hit', true)
  } catch (error) {
    core.setFailed(error.message)
  }
}

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
        const filePath = path.join(directory, file)
        const fileStats = await stat(filePath)

        if (fileStats.isFile() && fileStats.ctime < oneWeekAgo) {
          console.log(`Removing file: ${filePath}`)
          await unlink(filePath)
          totalSize -= fileStats.size
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
    core.error('Error removing old cache files')
    core.setFailed(error.message)
  }
}

module.exports = {
  save,
  cleanUp
}
