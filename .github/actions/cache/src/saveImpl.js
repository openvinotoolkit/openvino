const core = require('@actions/core');
const tar = require('tar');
const fs = require('fs/promises');
const path = require('path');
const { humanReadableFileSize, checkFileExists } = require('./utils');

/**
 * The main function for the action.
 * @returns {Promise<void>} Resolves when the action is complete.
 */
async function save() {
  try {
    const cacheRemotePath = core.getInput('cache-path', { required: true });
    const toCachePath = core.getInput('path', { required: true });
    const key = core.getInput('key', { required: true });

    core.debug(`cache-path: ${cacheRemotePath}`);
    core.debug(`path: ${toCachePath}`);
    core.debug(`key: ${key}`);

    if (!key) {
      core.warning(`Key ${key} is not specified.`);
      return;
    }

    const tarName = `${key}.cache`;
    const tarPath = path.join(cacheRemotePath, tarName);
    const tarNameTmp = `${key}.tmp`;
    const tarPathTmp = path.join(cacheRemotePath, tarNameTmp);

    if (await checkFileExists(tarPath)) {
      core.warning(`Cache file ${tarName} already exists`);
      return;
    }

    core.info(`Preparing cache archive ${tarName}`);
    tar.c(
      {
        gzip: true,
        file: tarName,
        cwd: toCachePath,
        sync: true
      },
      ['.']
    );
    const tarStat = await fs.stat(tarName);
    core.info(
      `Created cache tarball: ${tarName}, size: ${humanReadableFileSize(tarStat.size)}`
    );

    // remote cache directory may not be created yet
    if (!(await checkFileExists(cacheRemotePath))) {
      await fs.mkdir(cacheRemotePath);
    }

    core.info('Copying cache...');
    await fs.copyFile(tarName, tarPathTmp);
    // After copying is done, rename file
    await fs.rename(tarPathTmp, tarPath);
    core.info(`${tarName} copied to ${tarPath}`);

    core.setOutput('cache-file', tarName);
    core.setOutput('cache-hit', true);
  } catch (error) {
    core.setFailed(error.message);
  }
}

module.exports = {
  save
};
