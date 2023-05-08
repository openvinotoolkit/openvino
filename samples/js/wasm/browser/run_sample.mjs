import { printOVInfo } from './common/index.mjs';

const form = document.forms['form'];
const fieldset = document.querySelector('fieldset');

form.addEventListener('change', async () => {
  fieldset.disabled = true;

  const sampleName = form.elements.sample.value;

  await run(sampleName);

  fieldset.disabled = false;
});

async function run(sampleName) {
  const { default: sample } = await import(`./samples/${sampleName}.mjs`);

  await printOVInfo(openvinojs);
  console.log(`= Run sample: ${sampleName}`);
  await sample(openvinojs);
  console.log('= End\n');
}
