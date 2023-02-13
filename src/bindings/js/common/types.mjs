export const jsTypeByPrecisionMap = {
  int8: Int8Array,
  uint8: Uint8Array,
  uint8c: Uint8ClampedArray,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,

  float32: Float32Array,
  float64: Float64Array,

  int64: BigInt64Array,
  uint64: BigUint64Array,
};

export const heapLabelByTypeMap = {
  Int8Array: 'HEAP8',
  Uint8Array: 'HEAPU8',
  Uint8ClampedArray: 'HEAPU8',
  Int16Array: 'HEAP16',
  Uint16Array: 'HEAPU16',
  Int32Array: 'HEAP32',
  Uint32Array: 'HEAPU32',

  Float32Array: 'HEAPF32',
  Float64Array: 'HEAPF64',
}
