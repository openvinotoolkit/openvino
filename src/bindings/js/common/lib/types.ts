export type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type JSArrayType = 
  | Int8ArrayConstructor 
  | Uint8ArrayConstructor 
  // | Uint8ClampedArrayConstructor
  | Int16ArrayConstructor 
  | Uint16ArrayConstructor
  | Int32ArrayConstructor 
  | Uint32ArrayConstructor 
  | Float32ArrayConstructor 
  | Float64ArrayConstructor;

export enum PrecisionSupportedTypes {
  uint8 = 'uint8',
  int8 = 'int8',
  uint16 = 'uint16',
  int16 = 'int16',
  uint32 = 'uint32',
  int32 = 'int32',

  float32 = 'float32',
  float64 = 'float64',
};
export type PrecisionSupportedType = keyof typeof PrecisionSupportedTypes;

export interface IShape {
  dim: number,
  data: Uint32Array,
};

export interface ITensor {
  precision: PrecisionSupportedType,
  data: TypedArray,
  shape: IShape,
};

export interface IModel {
  infer(tensorOrDataArray: ITensor | number[], shape: IShape): Promise<ITensor>,
}
