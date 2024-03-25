const core = require('@actions/core')
const tar = require('tar')
const fs = require('fs')
const path = require('path')
const { humanReadableFileSize } = require('./utils')

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
    var tarNameTmp = `${key}.tmp`
    var tarPathTmp = path.join(cacheRemotePath, tarNameTmp)

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
    tarSize = fs.statSync(tarName).size
    core.info(
      `Created cache tarball: ${tarName}, size: ${humanReadableFileSize(tarSize)}`
    )

    // remote cache directory may not be created yet
    if (!fs.existsSync(cacheRemotePath)) {
      fs.mkdirSync(cacheRemotePath)
    }

    core.info('Copying cache...')
    fs.copyFileSync(tarName, tarPathTmp)
    // After copying is done, rename file
    fs.renameSync(tarPathTmp, tarPath)
    core.info(`${tarName} copied to ${tarPath}`)

    core.setOutput('cache-file', tarName)
    core.setOutput('cache-hit', true)
  } catch (error) {
    core.setFailed(error.message)
  }
}

module.exports = {
  save
}
