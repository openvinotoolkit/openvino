import loadModel from './node-model';
import createModule from 'openvinojs-common';

import type { ovNodeModule } from './types';

/* eslint-disable @typescript-eslint/no-var-requires */
const addon: ovNodeModule = require('../build/Release/ov_node_addon.node');

export default createModule(
  'node',
  loadModel,
  getVersionString,
  getDescriptionString
);
export { addon };

async function getVersionString(): Promise<string> {
  const str = 'Version';

  return str;
}

async function getDescriptionString(): Promise<string> {
  return addon.getDescriptionString();
}
