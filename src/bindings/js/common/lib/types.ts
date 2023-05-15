import Shape from './shape';
import Tensor from './tensor';

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
  u8 = 'u8',
  i8 = 'int8',
  u16 = 'u16',
  i16 = 'i16',
  u32 = 'u32',
  i32 = 'i32',

  f32 = 'f32',
  f64 = 'f64',
}
export type PrecisionSupportedType = keyof typeof PrecisionSupportedTypes;

export interface IShape {
  dim: number,
  data: Uint32Array,
}

export interface ITensor {
  precision: PrecisionSupportedType,
  data: TypedArray,
  shape: IShape,
}

export interface IModel {
  infer(tensor: ITensor): Promise<ITensor>;
  infer(dataArray: number[] | TypedArray, shape: IShape | number[])
    : Promise<ITensor>;
}

export interface ModelFiles {
  xml: string;
  bin: string;
}

export interface ModelNameAndPath {
  path: string;
  modelName: string;
}

export type LoadModelExternalType = {
  (
    path: string,
    shapeData: Shape | number[],
    layout: string
  ): Promise<IModel>,
  (
    filesPaths: ModelFiles,
    shapeData: Shape | number[],
    layout: string
  ): Promise<IModel>,
  (
    searchData: ModelNameAndPath,
    shapeData: Shape | number[],
    layout: string
  ): Promise<IModel>,
  (
    arg: ModelFiles | ModelNameAndPath | string,
    shapeData: Shape | number[],
    layout: string
  ): Promise<IModel>,
};

export type LoadModelInternalType = (
  xmlPath: string,
  binPath: string,
  shapeData: Shape | number[],
  layout: string
) => Promise<IModel>;

export interface IOpenVINOJSLibrary {
  loadModel: LoadModelExternalType,
  getVersionString: () => Promise<string>,
  getDescriptionString: () => Promise<string>,
  Shape: new (...dimensionsArray: number[] | [number[]]) => Shape,
  Tensor: new (
    precision: PrecisionSupportedType,
    data: number[] | TypedArray,
    shapeData: IShape | number[]
  ) => Tensor,
}

export type OpenVINOJSLibrary = Promise<IOpenVINOJSLibrary>;
