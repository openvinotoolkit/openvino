import { TypedArray, OVType, PrecisionSupportedType } from './types';

export interface OpenvinoModule {
  HEAP8: TypedArray,
  HEAPU8: TypedArray,
  HEAP16: TypedArray,
  HEAPU16: TypedArray,
  HEAP32: TypedArray,
  HEAPU32: TypedArray,
  HEAPF32: TypedArray,
  HEAPF64: TypedArray,
  _malloc: (amount: number) => number,
  _free: (heapPointer: number) => void,
  Shape: new (heapPointer: number, dimensions: number) => OriginalShape,
  Tensor: new (precision: PrecisionSupportedType, heapPointer: number, shape: OriginalShape) => OriginalTensor,
};

export interface OriginalShape {
  // constructor(heapPointer: number, dimensions: number);
  getDim(): number;
  getData(): number;
};

export interface OriginalShapeWrapper {
  obj: OriginalShape,
  free: () => void,
}

export interface OriginalTensor {
  getPrecision(): OVType;
  getShape(): OriginalShape;
  getData(): number;
}

export interface OriginalTensorWrapper {
  obj: OriginalTensor,
  free: () => void,
}
