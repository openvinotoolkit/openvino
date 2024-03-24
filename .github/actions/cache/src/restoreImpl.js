const core = require('@actions/core')
const fs = require('fs')
const path = require('path')
const tar = require('tar')
const os = require('os')

const { getSortedCacheFiles, humanReadableFileSize } = require('./utils')

/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function restore() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true })
    const cacheLocalPath = core.getInput('path', { required: true })
    const key = core.getInput('key', { required: true })
    const keysRestore = core
      .getInput('restore-keys', { required: false })
      .split('\n')
      .map(s => s.replace(/^!\s+/, '!').trim())
      .filter(x => x !== '')

    core.debug(`cache-path: ${cacheRemotePath}`)
    core.debug(`path: ${cacheLocalPath}`)
    core.debug(`key: ${key}`)
    core.debug(`restore-keys: ${keysRestore}`)

    var keyPattern = key
    if (keysRestore && keysRestore.length) {
      keyPattern = keysRestore.join('|')
    }

    core.info(`Looking for ${keyPattern} in ${cacheRemotePath}`)
    files = await getSortedCacheFiles(cacheRemotePath, keyPattern)

    if (files.length) {
      cacheFile = files[0]
      cacheSize = fs.statSync(path.join(cacheRemotePath, cacheFile)).size
      core.info(
        `Found cache file: ${cacheFile}, size: ${humanReadableFileSize(cacheSize)}`
      )

      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cache-'))
      // copy file to temp dir
      fs.copyFileSync(
        path.join(cacheRemotePath, cacheFile),
        path.join(tempDir, cacheFile)
      )
      core.info(`${cacheFile} was copied to ${tempDir}/${cacheFile}`)

      // extract
      if (!fs.existsSync(cacheLocalPath)) {
        fs.mkdirSync(cacheLocalPath)
      }
      core.info(`Extracting ${cacheFile} to ${cacheLocalPath}`)
      tar.x({
        file: path.join(tempDir, cacheFile),
        cwd: cacheLocalPath,
        sync: true
      })
      core.info(`Cache extracted to ${cacheLocalPath}`)

      core.setOutput('cache-file', cacheFile)
      core.setOutput('cache-hit', true)
    } else {
      core.warning(
        `Could not found any suitable cache files in ${cacheRemotePath} with key ${key}`
      )
      core.setOutput('cache-file', '')
      core.setOutput('cache-hit', false)
    }
  } catch (error) {
    // do not fail action if cache could not be restored
    core.warning(error.message)
  }
}

module.exports = {
  restore
}
