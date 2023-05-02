import { TypedArray } from 'openvinojs-common';

export interface ovNodeModule {
  Tensor: new (
    precision: string,
    shape: number[] | Uint32Array | Int32Array,
    tensor_data: TypedArray
  ) => NodeTensor,
  Model: new () => NodeModel,
  Shape: new (dimension: number, data: Uint32Array) => ShapeLite,
  getDescriptionString(): string
}

export interface NodeTensor {
  data: TypedArray;
  getData(): TypedArray;
  getPrecision(): string;
  getShape(): ShapeLite;
}

export interface ShapeLite {
  getDim(): number;
  getData(): number;
  shapeSize(): number;
}

export interface NodeModel {
  read_model(path: string): NodeModel;
  compile(device: string): NodeModel;
  infer(tensor: NodeTensor): NodeTensor;
}
