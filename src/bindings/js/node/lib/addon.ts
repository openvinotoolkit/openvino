interface Core {
  compileModel(model: Model, device: string): CompiledModel;
  readModelAsync(): Promise<Model>;
  readModel(modelPath: string, binPath?: string): Model;
}
interface CoreConstructor {
  new(): Core;
}

interface Model {
  outputs: Output[];
  inputs: Output[];

  output(nameOrId?: string | number): Output;
  getName(): string;
}

interface CompiledModel {
  outputs: Output[];
  inputs: Output[];
  output(nameOrId?: string | number): Output;
  input(nameOrId?: string | number): Output;
  createInferRequest(): InferRequest;
}

interface Tensor {
  // FIXME: now its only Float32Array
  data: number[];
  getPrecision(): element;
  // FIXME: change method return type, when remove Shape
  getShape(): Shape;
  // FIXME: now it returns Float32Array
  getData(): number[];
}
interface TensorConstructor {
  // FIXME: does empty constructor use?
  new(): Tensor;
  // FIXME: now tensorData can be only Float32Array
  new(type?: element, shape?: number[], tensorData?: number[]): Tensor;
}

interface Shape {
  data: number[];
  getData(): number[];
  shapeSize(): number;
  getDim(): number;
}
interface ShapeConstructor {
  // FIXME: does empty constructor use?
  new(): Shape;
  // FIXME: now tensorData can be only Float32Array
  new(dimensions: number, data: number[]): Shape;
}

interface InferRequest {
  // FIXME: are we going to add index parameter for this method?
  getOutputTensor(): Tensor;
  getOutputTensors(): Tensor[];
  getTensor(output: Output): Tensor;
  infer(inputData?: { [inputName: string]: Tensor }): void;
  setInputTensor(tensor: Tensor): void;
}

interface Output {
  anyName: string;
  shape: number[];
  // Constructor isn't available from JS side
  // constructor();
  toString(): string;
  getAnyName(): string;
  getShape(): Shape;
  // FIXME: These methods are not available from JS side for some reason
  // const ov::Node & ov::Node - the reason
  setNames(names: string[]): void;
  getNames(): string[];
}

interface PrePostProcessor {
  // FIXME: should we return this after build() call?
  build(): PrePostProcessor;
  setInputElementType(idx: number, type: element): PrePostProcessor;
  setInputModelLayout(layout: string[]): PrePostProcessor;
  // FIXME: are we going to add index parameter for this method?
  setInputTensorLayout(layout: string[]): PrePostProcessor;
  preprocessResizeAlgorithm(resizeAlgorithm: resizeAlgorithms)
    : PrePostProcessor;
  setInputTensorShape(shape: number[]): PrePostProcessor;
}
interface PrePostProcessorConstructor {
  new(model: Model): PrePostProcessor;
}

declare enum element {
  u8,
  u32,
  u16,
  i8,
  i64,
  i32,
  i16,
  f64,
  f32,
}

declare enum resizeAlgorithms {
  RESIZE_NEAREST,
  RESIZE_CUBIC,
  RESIZE_LINEAR,
}

export interface NodeAddon {
  Core: CoreConstructor,
  Tensor: TensorConstructor,
  Shape: ShapeConstructor,
  PrePostProcessor: PrePostProcessorConstructor,

  element: typeof element,
  resizeAlgorithms: typeof resizeAlgorithms,
}

export default
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../build/Release/ov_node_addon.node') as
    NodeAddon;
