import loadModel from './wasm-model';
import { getVersionString, getDescriptionString } from './wasm-model';

import createModule from 'openvinojs-common';

export default createModule(
  'wasm',
  loadModel,
  getVersionString,
  getDescriptionString
);
