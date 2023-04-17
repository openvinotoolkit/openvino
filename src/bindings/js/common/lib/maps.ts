import type { JSArrayType, PrecisionSupportedType } from './types.js';

export const jsTypeByPrecisionMap: { [Precision in PrecisionSupportedType]: JSArrayType } = {
  int8: Int8Array,
  uint8: Uint8Array,
  // uint8c: Uint8ClampedArray,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  float32: Float32Array,
  float64: Float64Array,
};
