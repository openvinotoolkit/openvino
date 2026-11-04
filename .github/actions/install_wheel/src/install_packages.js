import * as core from '@actions/core';
import { readdir } from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function getPythonVersion() {
  const { stdout } = await execAsync('python --version');
  const versionMatch = stdout.match(/Python (\d+)\.(\d+)\.(\d+)/);
  if (versionMatch) {
    return {
      major: versionMatch[1],
      minor: versionMatch[2],
      patch: versionMatch[3]
    };
  } else {
    throw new Error('Unable to detect Python version');
  }
}

async function installPackages(packages, localWheelDir, requirementsFiles) {
  core.debug(`Packages to install: ${packages}`);
  core.debug(`Local wheel directory: ${localWheelDir}`);
  core.debug(`Requirements files: ${requirementsFiles}`);

  const pythonVersion = await getPythonVersion();
  core.debug(`Detected Python version: ${JSON.stringify(pythonVersion)}`);

  // Resolve local wheels
  const localWheels = {};
  if (localWheelDir) {
    const entries = await readdir(localWheelDir);
    const wheels = entries
      .filter(entry => entry.endsWith('.whl'))
      .map(entry => path.join(localWheelDir, entry));
    core.debug(`Found wheels: ${wheels}`);
    for (const whl of wheels) {
      const packageName = path.basename(whl).split('-')[0];
      const wheelPythonVersion = path.basename(whl).match(/cp(\d{2,3})/);
      if (
        !wheelPythonVersion ||
        wheelPythonVersion[1] === `${pythonVersion.major}${pythonVersion.minor}`
      ) {
        localWheels[packageName] = whl;
      }
    }
  }
  core.debug(`Resolved local wheels: ${JSON.stringify(localWheels)}`);

  // Collect wheel paths
  const wheelPaths = [];
  for (const pkg of packages) {
    const packageName = pkg.split('[')[0];
    if (localWheels[packageName]) {
      const wheelPath = localWheels[packageName];
      wheelPaths.push(`"${wheelPath}${pkg.slice(packageName.length)}"`);
    } else {
      core.setFailed(`Package ${pkg} not found locally.`);
      return;
    }
  }
  core.debug(`Collected wheel paths: ${wheelPaths}`);

  // Collect requirements files
  const requirementsArgs = requirementsFiles.map(reqFile => `-r ${reqFile}`);
  core.debug(`Requirements arguments: ${requirementsArgs}`);

  // Install all wheels and requirements in one command
  const installArgs = [...wheelPaths, ...requirementsArgs];
  if (installArgs.length > 0) {
    core.debug(`Installing packages with arguments: ${installArgs.join(' ')}`);
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        core.debug(`Attempt ${attempt} of 3`);
        const { stdout, stderr } = await execAsync(
          `python -m pip install ${installArgs.join(' ')}`,
          {
            stdio: 'inherit'
          }
        );
        if (stdout) {
          core.debug('stdout:', stdout);
        }
        if (stderr) {
          core.error('stderr:', stderr);
        }
        break;
      } catch (error) {
        core.error(`Attempt ${attempt + 1} failed:`, error.message);
        if (attempt === 2) {
          throw error;
        }
        const sleepTime = Math.pow(2, attempt) * 1000;
        core.debug(`Waiting ${sleepTime / 1000} seconds before retry...`);
        await new Promise(resolve => setTimeout(resolve, sleepTime));
      }
    }
  }
}

async function run() {
  try {
    const packagesInput = core.getInput('packages');
    const localWheelDir = core.getInput('local_wheel_dir') || null;
    const requirementsInput = core.getInput('requirements_files') || '';
    const packages = packagesInput.split(';');
    const requirementsFiles = requirementsInput
      .split(';')
      .filter(Boolean)
      .map(reqFile => path.normalize(reqFile));
    const normalizedLocalWheelDir = localWheelDir
      ? path.normalize(localWheelDir)
      : null;
    await installPackages(packages, normalizedLocalWheelDir, requirementsFiles);
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
