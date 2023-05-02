import type { JSArrayType, PrecisionSupportedType } from './types';

export const jsTypeByPrecisionMap
: { [Precision in PrecisionSupportedType]: JSArrayType } = {
  i8: Int8Array,
  u8: Uint8Array,
  // u8c: Uint8ClampedArray,
  i16: Int16Array,
  u16: Uint16Array,
  i32: Int32Array,
  u32: Uint32Array,
  f32: Float32Array,
  f64: Float64Array,
};
