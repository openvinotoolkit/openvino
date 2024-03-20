const core = require('@actions/core')
const tar = require('tar')
const fs = require('fs')
const path = require('path')

/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function save() {
  try {
    const cachePath = core.getInput('cache_path', { required: true })
    const toCachePath = core.getInput('path', { required: true })
    const key = core.getInput('key', { required: true })

    core.debug(`cache_path: ${cachePath}`)
    core.debug(`path: ${toCachePath}`)
    core.debug(`key: ${key}`)

    if (!key) {
      core.warning(`Key ${key} is not specified.`)
      return
    }

    var tarName = `${key}.cache`
    var tarPath = path.join(cachePath, tarName)

    tar.c(
      {
        gzip: true,
        file: tarName,
        cwd: toCachePath,
        sync: true
      },
      ['.']
    )

    fs.copyFileSync(tarName, tarPath)
    core.info(`${tarName} was copied to ${tarPath}`)
    core.setOutput('cache-file', tarName)
    core.setOutput('cache-hit', true)
  } catch (error) {
    core.setFailed(error.message)
  }
}

module.exports = {
  save
}
