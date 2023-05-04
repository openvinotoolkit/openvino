import openvinoWASM from '../bin/openvino_wasm';

import type { OpenvinoWASMModule } from './types';

let initializedOVWASMModule: OpenvinoWASMModule;

export default async function getOVWASM(): Promise<OpenvinoWASMModule> {
  if (!initializedOVWASMModule) initializedOVWASMModule = await openvinoWASM();

  return initializedOVWASMModule;
}
