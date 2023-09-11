const os = require('os');
const path = require('path');
const fs = require('node:fs/promises');
const decompress = require('decompress');
const { createWriteStream } = require('node:fs');

const packageJson = require('../package.json');

const codeENOENT = 'ENOENT';

main();

async function main() {
  const osInfo = await detectOS();
  const modulePath = packageJson.binary['module_path'];

  const isRuntimeDirExists = await checkDirExistence(modulePath);

  if (isRuntimeDirExists && !process.argv.includes('-f')) {
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
    + packageName;

  try {
    await downloadRuntime(binaryUrl);
  } catch (err) {
    console.log(`Runtime fetch failed. Reason ${err}`);

    if (err instanceof Error) throw err;

    return;
  }

  console.log('Ready');
}

async function detectOS() {
  const platform = os.platform();

  if (!['win32', 'linux'].includes(platform)) {
    console.error(`Platform '${platform}' doesn't support`);
    process.exit(1);
  }

  const platformMapping = {
    win32: {
      os: 'windows',
      letter: 'w',
      extension: 'zip',
    },
    linux: {
      letter: 'l',
      extension: 'tgz',
    }
  };

  if (platform === 'linux') {
    const osReleaseData = await fs.readFile('/etc/os-release', 'utf8');
    const os = osReleaseData.includes('Ubuntu 22')
      ? 'ubuntu22'
      : osReleaseData.includes('Ubuntu 20')
      ? 'ubuntu20'
      : osReleaseData.includes('Ubuntu 18')
      ? 'ubuntu18'
      : null;

    if (!os) {
      console.error('Cannot detect your OS');
      process.exit(1);
    }

    platformMapping.linux.os = os;
  }

  const arch = os.arch();

  if (!['arm64', 'x64'].includes(arch)) {
    console.error(`Architecture '${arch}' doesn't support`);
    process.exit(1);
  }

  const archMapping = {
    arm64: 'arm64',
    x64: 'x86_64',
  };

  return { platform, arch: archMapping[arch], ...platformMapping[platform] };
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

async function downloadRuntime(uri, opts = {}) {
  const fetch = (await import('node-fetch')).default;

  console.log('GET', uri);

  // Try getting version info from the currently running npm.
  const envVersionInfo = process.env['npm_config_user_agent']
    || `node ${process.version}`;

  const sanitized = uri.replace('+', '%2B');
  const requestOpts = {
    uri: sanitized,
    headers: { 'User-Agent': `openvinojs-node (${envVersionInfo})` },
    'follow_max': 10,
  };

  if (opts.cafile) {
    try {
      requestOpts.ca = fs.readFileSync(opts.cafile);
    } catch (e) {
      return callback(e);
    }
  } else if (opts.ca) {
    requestOpts.ca = opts.ca;
  }

  const proxyUrl = opts.proxy || process.env.http_proxy ||
    process.env.HTTP_PROXY || process.env.npm_config_proxy;
  let agent;
  if (proxyUrl) {
    const ProxyAgent = require('https-proxy-agent');

    agent = new ProxyAgent(proxyUrl);
    console.log(`Proxy agent configured using: '${proxyUrl}'`);
  }

  const res = await fetch(sanitized, { agent });

  if (!res.ok)
    throw new Error(`Response status ${res.status} ${res.statusText} `
      + `on ${sanitized}`);

  const filename = path.basename(uri);
  const tmpPath = path.resolve(__dirname, '..', 'temp');
  const fullPath = path.resolve(tmpPath, filename);

  try {
    await fs.rm(fullPath);
    await fs.rmdir(tmpPath);
  } catch(err) {
    if (err.code !== codeENOENT) throw err;
  }

  await fs.mkdir(tmpPath, {  });

  const fileStream = createWriteStream(fullPath, { flags: 'wx' });

  return new Promise((resolve, reject) => {
    let error = null;
    const dataStream = res.body

    console.log('Downloading openvino runtime archive');

    dataStream.pipe(fileStream).on('error', err => error = err);
    dataStream.on('end', async () => {
      if (error || (res.status !== 200))
        return reject(error || `Status: ${res.status}, ${res.statusText}`);

      console.log('Downloaded');

      const runtimeDir = path.resolve(__dirname, '..', 'ov_runtime');

      console.log('Uncompressing...');
      await decompress(fullPath, runtimeDir, { strip: 1 });
      await fs.rm(fullPath);
      await fs.rmdir(tmpPath);
      console.log('The archive was successfully uncompressed');

      resolve();
    });
  });
}
