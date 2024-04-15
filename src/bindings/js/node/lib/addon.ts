type SupportedTypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

type elementTypeString =
  | 'u8'
  | 'u32'
  | 'u16'
  | 'u64'
  | 'i8'
  | 'i64'
  | 'i32'
  | 'i16'
  | 'f64'
  | 'f32';

interface Core {
  compileModel(
    model: Model,
    deviceName: string,
    config?: { [option: string]: string }
  ): Promise<CompiledModel>;
  compileModelSync(
    model: Model,
    deviceName: string,
    config?: { [option: string]: string }
  ): CompiledModel;
  readModel(modelPath: string, weightsPath?: string): Promise<Model>;
  readModel(
    modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Promise<Model>;
  readModelSync(modelPath: string, weightsPath?: string): Model;
  readModelSync(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Model;
  importModelSync(modelStream: Buffer, device: string): CompiledModel;
  getAvailableDevices(): string[];
  getVersions(deviceName: string): {
    [deviceName: string]: {
      buildNumber: string,
      description: string,
    },
  };
  setProperty(props: { [key: string]: string | number | boolean }): void;
  setProperty(
    deviceName: string,
    props: { [key: string]: string | number | boolean },
  ): void;
  getProperty(propertyName: string): string | number | boolean,
  getProperty(
    deviceName: string,
    propertyName: string,
  ): string | number | boolean;
  addExtension(libraryPath: string): void;
}
interface CoreConstructor {
  new(): Core;
}

interface Model {
  outputs: Output[];
  inputs: Output[];
  output(nameOrId?: string | number): Output;
  input(nameOrId?: string | number): Output;
  getName(): string;
  isDynamic(): boolean;
}

interface CompiledModel {
  outputs: Output[];
  inputs: Output[];
  output(nameOrId?: string | number): Output;
  input(nameOrId?: string | number): Output;
  createInferRequest(): InferRequest;
  exportModelSync(): Buffer;
}

interface Tensor {
  data: number[];
  getElementType(): element;
  getShape(): number[];
  getData(): number[];
  getSize(): number;
}
interface TensorConstructor {
  new(type: element | elementTypeString,
      shape: number[],
      tensorData?: number[] | SupportedTypedArray): Tensor;
}

interface InferRequest {
  setTensor(name: string, tensor: Tensor): void;
  setInputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void;
  setOutputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void;
  getTensor(nameOrOutput: string | Output): Tensor;
  getInputTensor(idx?: number): Tensor;
  getOutputTensor(idx?: number): Tensor;
  infer(inputData?: { [inputName: string]: Tensor | SupportedTypedArray}
    | Tensor[] | SupportedTypedArray[]): { [outputName: string] : Tensor};
  inferAsync(inputData: { [inputName: string]: Tensor}
    | Tensor[] ): Promise<{ [outputName: string] : Tensor}>;
  getCompiledModel(): CompiledModel;
}

type Dimension = number | [number, number];

interface Output {
  anyName: string;
  shape: number[];
  toString(): string;
  getAnyName(): string;
  getShape(): number[];
  getPartialShape(): PartialShape;
}

interface InputTensorInfo {
  setElementType(elementType: element | elementTypeString ): InputTensorInfo;
  setLayout(layout: string): InputTensorInfo;
  setShape(shape: number[]): InputTensorInfo;
}

interface OutputTensorInfo {
  setElementType(elementType: element | elementTypeString ): InputTensorInfo;
  setLayout(layout: string): InputTensorInfo;
}
interface PreProcessSteps {
  resize(algorithm: resizeAlgorithm | string): PreProcessSteps;
}

interface InputModelInfo {
  setLayout(layout: string): InputModelInfo;
}

interface InputInfo {
  tensor(): InputTensorInfo;
  preprocess(): PreProcessSteps;
  model(): InputModelInfo;
}

interface OutputInfo {
  tensor(): OutputTensorInfo;
}

interface PrePostProcessor {
  build(): PrePostProcessor;
  input(idxOrTensorName?: number | string): InputInfo;
  output(idxOrTensorName?: number | string): OutputInfo;
}
interface PrePostProcessorConstructor {
  new(model: Model): PrePostProcessor;
}

interface PartialShape {
  isStatic(): boolean;
  isDynamic(): boolean;
  toString(): string;
  getDimensions(): Dimension[];
}
interface PartialShapeConstructor {
  new(shape: string): PartialShape;
}

declare enum element {
  u8,
  u32,
  u16,
  u64,
  i8,
  i16,
  i32,
  i64,
  f32,
  f64,
}

declare enum resizeAlgorithm {
  RESIZE_NEAREST,
  RESIZE_CUBIC,
  RESIZE_LINEAR,
}

export interface NodeAddon {
  Core: CoreConstructor,
  Tensor: TensorConstructor,
  PartialShape: PartialShapeConstructor,

  preprocess: {
    resizeAlgorithm: typeof resizeAlgorithm,
    PrePostProcessor: PrePostProcessorConstructor,
  },
  element: typeof element,
}

export default
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../bin/ov_node_addon.node') as
    NodeAddon;
