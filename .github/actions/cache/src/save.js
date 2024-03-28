const core = require('@actions/core');
const { save } = require('./saveImpl');
const { cleanUp } = require('./cleanupImpl');

const saveAlways = core.getInput('save-always', { required: false });
const cleanUpAlways = core.getInput('cleanup-always', { required: false });

if (saveAlways === 'true') {
  save();
}

if (cleanUpAlways === 'true') {
  cleanUp();
}
