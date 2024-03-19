const core = require('@actions/core')
const { log, error } = require('console')
const fs = require('fs')
const path = require('path')
const tar = require('tar')

const { getSortedCacheFiles } = require('./cache')

/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function restore() {
  try {
    const cacheRemotePath = core.getInput('cache_path', { required: true })
    const cacheLocalPath = core.getInput('path', { required: true })
    const key = core.getInput('key', { required: true })

    log(cacheRemotePath)
    log(cacheLocalPath)
    log(key)

    // Debug logs are only output if the `ACTIONS_STEP_DEBUG` secret is true
    core.debug(`Looking for ${key} in ${cacheRemotePath}`)
    files = await getSortedCacheFiles(cacheRemotePath).catch(error)
    if (files.length) {
      cacheFile = files[0]

      // copy file to local fs
      if (!fs.existsSync(cacheLocalPath)) {
        fs.mkdirSync(cacheLocalPath)
      }
      fs.copyFileSync(
        path.join(cacheRemotePath, cacheFile),
        path.join(cacheLocalPath, cacheFile)
      )
      log(`${cacheFile} was copied to ${cacheLocalPath}/${cacheFile}`)

      // extract
      tar.x({
        file: path.join(cacheLocalPath, cacheFile),
        cwd: cacheLocalPath,
        sync: true
      })

      core.setOutput('cache-file', cacheFile)
      core.setOutput('cache-hit', true)
    } else {
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
