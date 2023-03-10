import { OpenvinoModule, OriginalShapeWrapper, OriginalTensorWrapper } from './ov-module.js';

export type OVType =
  | 'uint8_t'
  | 'int8_t'
  | 'uint16_t'
  | 'int16_t'
  | 'uint32_t'
  | 'int32_t'
  | 'float'
  | 'double';

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

export type HEAPType = 
  | 'HEAP8'
  | 'HEAPU8'
  | 'HEAP16'
  | 'HEAPU16'
  | 'HEAP32'
  | 'HEAPU32'
  | 'HEAPF32'
  | 'HEAPF64';

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
  convert(ov: OpenvinoModule): OriginalShapeWrapper,
};

export interface ITensor {
  precision: PrecisionSupportedType,
  data: TypedArray,
  shape: IShape,
  convert(ov: OpenvinoModule): OriginalTensorWrapper,
};

export type SessionEnvironment = 'nodejs' | 'browser';

export interface IModel {
  // new (ov: OpenvinoModule, originalModel: OriginalModel): void,
  infer(tensor: ITensor): Promise<ITensor>,
}

export interface ISession {
  _ov: OpenvinoModule,
  _env: SessionEnvironment,

  getVersionString(): string,
  getDescriptionString(): string,

  // new (ov: OpenvinoModule, environment?: SessionEnvironment): ISession,
  // loadModel(xmlPath: string, binPath: string, shape: IShape, layout: string): Promise<IModel>
  loadModel(xmlData: Uint8Array, binData: Uint8Array, shapeData: number[] | IShape, layout: string): Promise<IModel>
}
