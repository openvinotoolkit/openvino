import type { TypedArray, OVType, PrecisionSupportedType } from './types.js';

interface WASMFilesystem {
  open(filename: string, flags: string): string,
  write(stream: string, data: Uint8Array, position: number, length: number, from: number): void,
  close(stream: string): void,
}

export interface OpenvinoModule {
  FS: WASMFilesystem,
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
  Session: new (xmlFilename: string, binFilename: string, originalShapeObj: OriginalShape, layout: string) => OriginalModel,
  getVersionString(): string,
  getDescriptionString(): string,
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

export interface OriginalSession {
  loadModel(): OriginalModel,
}

export interface OriginalModel {
  infer(tensor: OriginalTensor): OriginalTensor,
}
