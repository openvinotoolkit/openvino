const os = require('os');
const path = require('path');
const fs = require('node:fs/promises');
const decompress = require('decompress');
const { createWriteStream } = require('node:fs');
const { HttpsProxyAgent } = require('https-proxy-agent');

const packageJson = require('../package.json');

const codeENOENT = 'ENOENT';

main();

async function main() {
  const osInfo = await detectOS();
  const isForce = process.argv.includes('-f');
  const modulePath = packageJson.binary['module_path'];

  const isRuntimeDirExists = await checkDirExistence(modulePath);

  if (isRuntimeDirExists && !isForce) {
    if (process.argv.includes('--ignore-if-exists')) {
      console.log(`Directory '${modulePath}' exists, skip runtime init `
        + 'because \'--ignore-if-exists\' flag passed');
      return;
    }

    console.error(`Directory '${modulePath}' already exists, to force `
      + `runtime installation run 'npm run download_runtime -- -f'`);
    process.exit(1);
  }

  const originalPackageName = packageJson.binary['package_name'];

  let packageName = originalPackageName.replace('{letter}', osInfo.letter);
  packageName = packageName.replace('{os}', osInfo.os);
  packageName = packageName.replace('{extension}', osInfo.extension);
  packageName = packageName.replace('{arch}', osInfo.arch);
  packageName = packageName.replace('{version}', packageJson.binary.version);

  const binaryUrl = packageJson.binary.host + packageJson.binary['remote_path']
    + `${osInfo.dir}/` + packageName;

  try {
    await fetchRuntime(binaryUrl);
  } catch (err) {
    console.log(`Runtime fetch failed. Reason ${err}`);

    if (err instanceof Error) throw err;

    return;
  }

  console.log('Ready');
}

async function detectOS() {
  const platform = os.platform();

  if (!['win32', 'linux', 'darwin'].includes(platform)) {
    console.error(`Platform '${platform}' doesn't support`);
    process.exit(1);
  }

  const platformMapping = {
    win32: {
      os: 'windows',
      dir: 'windows',
      letter: 'w',
      extension: 'zip',
    },
    linux: {
      letter: 'l',
      dir: 'linux',
      extension: 'tgz',
    },
    darwin: {
      letter: 'm',
      dir: 'macos',
      extension: 'tgz',
    },
  };

  const arch = os.arch();

  if (!['arm64', 'armhf', 'x64'].includes(arch)) {
    console.error(`Architecture '${arch}' doesn't support`);
    process.exit(1);
  }

  const archMapping = {
    arm64: 'arm64',
    armhf: 'armhf',
    x64: 'x86_64',
  };

  let osVersion = null;
  switch(platform) {
    case 'linux':
      const osReleaseData = await fs.readFile('/etc/os-release', 'utf8');

      osVersion = osReleaseData.includes('Ubuntu 22')
        ? 'ubuntu22'
        : osReleaseData.includes('Ubuntu 20')
        ? 'ubuntu20'
        : osReleaseData.includes('Ubuntu 18')
        ? 'ubuntu18'
        : ['arm64', 'armhf'].includes(arch)
          && osReleaseData.includes('ID=debian')
        ? 'debian9'
        : null;

      break;

    case 'darwin':
      const [major, _, __] = os.release().split('.');

      // os.release() returns not the macOS release but the Darwin release:
      // https://en.wikipedia.org/wiki/Darwin_(operating_system)#Release_history - mapping could be found here
      // in the form of MAJOR.MINOR.PATCH

      osVersion = major === '19'
        ? 'macos_10_15'
        : major === '20'
        ? 'macos_11_0'
        : null;

      break;

    case 'win32':
      osVersion = true;

      break;
  }

  if (!osVersion) {
    console.error('Cannot detect your OS');
    process.exit(1);
  }

  return {
    platform,
    os: osVersion,
    arch: archMapping[arch],
    ...platformMapping[platform]
  };
}

async function checkDirExistence(pathToDir) {
  try {
    await fs.access(pathToDir);

    return true;
  }
  catch (err) {
    if (err.code !== codeENOENT) throw err;

    return false;
  }
}

async function fetchRuntime(uri) {
  const filename = path.basename(uri);
  const tmpPath = path.resolve(__dirname, '..', 'temp');
  const fullPath = path.resolve(tmpPath, filename);
  const runtimeDir = path.resolve(__dirname, '..', 'ov_runtime');

  try {
    await fs.rm(tmpPath, { recursive: true, force: true });
  } catch(err) {
    if (err.code !== codeENOENT) throw err;
  }

  await fs.mkdir(tmpPath);
  console.log('Downloading openvino runtime archive...');
  await downloadFile(uri, filename, tmpPath);
  console.log('Downloaded');

  console.log('Uncompressing...');
  try {
    await fs.rm(runtimeDir, { recursive: true, force: true });
  } catch(err) {
    if (err.code !== codeENOENT) throw err;
  }
  await decompress(fullPath, runtimeDir, { strip: 1 });
  await fs.rm(tmpPath, { recursive: true, force: true });
  console.log('The archive was successfully uncompressed');
}

function downloadFile(url, filename, destination) {
  const { env } = process;
  const timeout = 5000;
  const fullPath = path.resolve(destination, filename);
  const file = createWriteStream(fullPath);
  const protocolString = new URL(url).protocol === 'https:' ? 'https' : 'http';
  const module = require(`node:${protocolString}`);
  const proxyUrl = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  let agent;

  if (proxyUrl) {
    agent = new HttpsProxyAgent(proxyUrl);
    console.log(`Proxy agent configured using: '${proxyUrl}'`);
  }

  return new Promise((resolve, reject) => {
    file.on('error', e => {
      reject(`Error oppening file stream: ${e}`);
    });

    const getRequest = module.get(url, { agent }, res => {
      const { statusCode } = res;

      if (statusCode !== 200)
        return reject(`Server returns status code: ${statusCode}`);

      res.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`File successfully stored at '${fullPath}'`);
        resolve();
      });
    });

    getRequest.on('error', e => {
      reject(`Error sending request: ${e}`);
    });

    getRequest.setTimeout(timeout, () => {
      getRequest.destroy();
      reject(`Request timed out after ${timeout}`);
    });
  });
}
