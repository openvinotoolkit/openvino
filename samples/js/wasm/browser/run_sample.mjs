const DEFAULT_SAMPLE = 'classification';
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

  console.log(`= Run sample: ${sampleName}`);
  await sample(openvinojs);
  console.log('= End');
}
