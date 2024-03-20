const core = require('@actions/core')
const fs = require('fs')
const path = require('path')
const tar = require('tar')
const os = require('os')

const { getSortedCacheFiles } = require('./cache')

/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function restore() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true })
    const cacheLocalPath = core.getInput('path', { required: true })
    const key = core.getInput('key', { required: true })
    var keysRestore = core
      .getInput('restore-keys', { required: false })
      .split('\n')
      .map(s => s.replace(/^!\s+/, '!').trim())
      .filter(x => x !== '')

    core.debug(`cache-path: ${cacheRemotePath}`)
    core.debug(`path: ${cacheLocalPath}`)
    core.debug(`key: ${key}`)
    core.debug(`restore-keys: ${keysRestore}`)

    if (!keysRestore || !keysRestore.length) {
      keysRestore = [key]
    }

    cacheFile = ''
    for (var i = 0; i < keysRestore.length; i++) {
      var keyR = keysRestore[i]
      core.info(`Looking for ${keyR} in ${cacheRemotePath}`)
      files = await getSortedCacheFiles(cacheRemotePath, keyR)
      if (files.length) {
        cacheFile = files[0]
        cacheSize = fs.statSync(path.join(cacheRemotePath, cacheFile)).size
        core.info(`Found cache file: ${cacheFile}, size: ${cacheSize}`)
        break
      }
    }

    if (cacheFile) {
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
    // Fail the workflow run if an error occurs
    core.setFailed(error.message)
  }
}

module.exports = {
  restore
}
