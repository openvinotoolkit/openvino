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
  | 'f32'
  | 'string';

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
  importModelSync(
    modelStream: Buffer,
    device: string,
    props: { [key: string]: string | number | boolean }
  ): CompiledModel;
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
  getOutputSize(): number;
  setFriendlyName(name: string): void;
  getFriendlyName(): string;
  getOutputShape(): number[];
}

/**
 * CompiledModel represents Model that is compiled for a specific device
 * by applying multiple optimization transformations,
 * then mapping to compute kernels.
 */
interface CompiledModel {
  /** Gets all inputs of a compiled model. */
  inputs: Output[];
  /** Gets all outputs of a compiled model. */
  outputs: Output[];
  /**
   * Creates an inference request object used to infer the compiled model.
   * @return {InferRequest}
   */
  createInferRequest(): InferRequest;
  /**
   * Exports the compiled model to binary data.
   * @remarks
   * The exported model can be imported via the {@link Core.importModelSync}.
   * @return {Buffer} The binary data that contains this compiled model.
   */
  exportModelSync(): Buffer;
  /**
   * Gets a single output of a compiled model.
   * If a model has more than one output, this method throws an exception.
   * @returns {Output} A compiled model output.
   */
  output(): Output;
  /**
   * Gets output of a compiled model identified by an index.
   * @param index An output tensor index.
   * @returns {Output} A compiled model output.
   */
  output(index: number): Output;
  /**
   * Gets output of a compiled model identified by a tensorName.
   * @param name An output tensor name.
   * @returns {Output} A compiled model output.
   */
  output(name: string): Output;
  /**
   * Gets a single input of a compiled model.
   * If a model has more than one input, this method throws an exception.
   * @returns {Output} A compiled model input.
   */
  input(): Output;
  /**
   * Gets input of a compiled model identified by an index.
   * @param index An input tensor index.
   * @returns {Output} A compiled model input.
   */
  input(index: number): Output;
  /**
   * Gets input of a compiled model identified by a tensorName.
   * @param name An input tensor name.
   * @returns {Output} A compiled model input.
   */
  input(name: string): Output;

}

interface Tensor {
  data: SupportedTypedArray;
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
  string,
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
