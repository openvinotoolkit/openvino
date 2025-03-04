const { join } = require('node:path');

const BinaryManager = require('./lib/binary-manager');
const packageJson = require('../package.json');

if (require.main === module) main();

async function main() {
  if (!BinaryManager.isCompatible()) process.exit(1);

  const force = process.argv.includes('-f') || process.argv.includes('--force');
  const ignoreIfExists = process.argv.includes('-i')
    || process.argv.includes('--ignore-if-exists');

  const { env } = process;
  const proxy = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  await BinaryManager.prepareBinary(
    join(__dirname, '..'),
    packageJson.binary.version || packageJson.version,
    packageJson.binary,
    { force, ignoreIfExists, proxy },
  );
}
