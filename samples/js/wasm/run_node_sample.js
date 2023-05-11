const openvinojs = require('openvinojs-wasm');

const DEFAULT_SAMPLE = 'classification';
const sample = process.argv[2] || DEFAULT_SAMPLE;

run(sample);

async function run(sample) {
  const { printOVInfo } = await import('../common/index.mjs');
  const sampleFilename = `../samples/${sample}.mjs`;

  await printOVInfo(openvinojs);
  console.log(`= Run sample: ${sampleFilename}`);
  const { default: runSample } = await import(sampleFilename);
  await runSample(openvinojs);
  console.log('= End\n');
}
