import * as core from '@actions/core';
import { save } from './saveImpl.js';
import { cleanUp } from './cleanupImpl.js';

const saveAlways = core.getInput('save-always', { required: false });
const cleanUpAlways = core.getInput('cleanup-always', { required: false });

if (saveAlways === 'true') {
  save();
}

if (cleanUpAlways === 'true') {
  cleanUp();
}
