import * as core from '@actions/core';
import { save } from './saveImpl.js';
import { cleanUp } from './cleanupImpl.js';

const cleanUpAlways = core.getInput('cleanup-always', { required: false });

await save();

if (cleanUpAlways === 'true') {
  await cleanUp();
}
