import { JSArrayType } from 'openvinojs-common';
import { HEAPType } from './types';

export const heapLabelByArrayTypeMap
: { [ArrayType in keyof JSArrayType as string]: HEAPType } = {
  Int8Array: 'HEAP8',
  Uint8Array: 'HEAPU8',
  // Uint8ClampedArray: 'HEAPU8',
  Int16Array: 'HEAP16',
  Uint16Array: 'HEAPU16',
  Int32Array: 'HEAP32',
  Uint32Array: 'HEAPU32',
  Float32Array: 'HEAPF32',
  Float64Array: 'HEAPF64',
};
