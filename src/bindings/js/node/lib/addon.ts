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

/**
 * Core represents OpenVINO runtime Core entity.
 *
 * User applications can create several Core class instances,
 * but in this case, the underlying plugins
 * are created multiple times and not shared between several Core instances.
 * The recommended way is to have a single Core instance per application.
 */
interface Core {
  /**
   * Registers extensions to a Core object.
   * @param libraryPath Path to the library with ov::Extension.
   */
  addExtension(libraryPath: string): void;
  /**
   * Asynchronously creates a compiled model from a source {@link Model} object.
   *
   * Users can create as many compiled models as they need and use them
   * simultaneously (up to the limitation of the hardware resources).
   * @param model {@link Model} object acquired from {@link Core.readModel}
   * @param deviceName Name of device to load model to.
   * @param config Object with key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  compileModel(
    model: Model,
    deviceName: string,
    config?: { [propertyName: string]: string }
  ): Promise<CompiledModel>;
  /**
   * Asynchronously reads a model and creates a compiled model
   * from the IR/ONNX/PDPD file.
   *
   * This can be more efficient
   * than using {@link Core.readModel} + core.compileModel(Model) flow
   * especially for cases when caching is enabled and a cached model is
   * available. Users can create as many compiled models as they need and use
   * them simultaneously (up to the limitation of the hardware resources).
   * @param modelPath Path to a model.
   * @param deviceName Name of a device to load a model to.
   * @param config Object with key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  compileModel(
    modelPath: string,
    deviceName: string,
    config?: { [propertyName: string]: string }
  ): Promise<CompiledModel>;
  /**
   * Synchronous version of {@link Core.compileModel}.
   * Creates a compiled model from a source model object.
   */
  compileModelSync(
    model: Model,
    deviceName: string,
    config?: { [propertyName: string]: string }
  ): CompiledModel;
  /**
   * Synchronous version of {@link Core.compileModel}.
   * Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
   */
  compileModelSync(
    modelPath: string,
    deviceName: string,
    config?: { [propertyName: string]: string }
  ): CompiledModel;
  /**
   * Returns devices available for inference.
   * Core objects go over all registered plugins.
   * @returns A list of devices. The devices are returned as: CPU, GPU.0,
   * GPU.1, NPU… If there is more than one device of a specific type, they are
   * enumerated with .# suffix. Such enumerated devices can later be used
   * as a device name in all Core methods like compile_model, query_model,
   * set_property and so on.
   */
  getAvailableDevices(): string[];
  /**
   * Gets properties dedicated to device behaviour.
   * @param propertyName Property name.
   */
  getProperty(propertyName: string): string | number | boolean;

  /**
   * Gets properties dedicated to device behaviour.
   * @param deviceName Name of a device to get a properties
   * @param propertyName Property name.
   */
  getProperty(
    deviceName: string,
    propertyName: string,
  ): string | number | boolean;
  /**
   * Returns device plugins version information.
   * @param deviceName Device name to identify a plugin.
   */
  getVersions(deviceName: string): {
    [deviceName: string]: {
      buildNumber: string,
      description: string,
    },
  };
  /**
   * Imports a compiled model from a previously exported one.
   * @param modelStream Input stream, containing a model previously exported,
   * using {@link CompiledModel.exportModelSync} method.
   * @param device Name of a device to import a compiled model for.
   * Note, if the device name was not used to compile the original mode,
   * an exception is thrown.
   * @param config Object with key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  importModelSync(
    modelStream: Buffer,
    device: string,
    config?: { [key: string]: string | number | boolean }
  ): CompiledModel;
  /**
   * Reads models from IR / ONNX / PDPD / TF and TFLite formats.
   * @param modelPath A path to a model
   * in IR / ONNX / PDPD / TF and TFLite format.
   * @param weightsPath A path to a data file For IR format (.bin): if the path
   * is empty, it tries to read a bin file with the same name as xml and if
   * the bin file with the same name was not found, loads IR without weights.
   * For ONNX format (.onnx): weights parameter is not used.
   * For PDPD format (.pdmodel) weights parameter is not used.
   * For TF format (.pb) weights parameter is not used.
   * For TFLite format (*.tflite) weights parameter is not used.
   */
  readModel(modelPath: string, weightsPath?: string): Promise<Model>;

  /**
   * Reads models from IR / ONNX / PDPD / TF and TFLite formats.
   * @param modelBuffer Binary data with model
   * in IR / ONNX / PDPD / TF and TFLite format.
   * @param weightsBuffer Binary data with tensor’s data.
   */
  readModel(
    modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Promise<Model>;
  /**
   * Synchronous version of {@link Core.readModel}.
   * Reads models from IR / ONNX / PDPD / TF and TFLite formats.
   */
  readModelSync(modelPath: string, weightsPath?: string): Model;
  /**
   * Synchronous version of {@link Core.readModel}.
   * Reads models from IR / ONNX / PDPD / TF and TFLite formats.
   */
  readModelSync(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Model;
  /**
   * Sets properties.
   * @param properties Object with pairs: property name - property value
   */
  setProperty(properties: { [key: string]: string | number | boolean }): void;
  /**
   * Sets properties for the device.
   * @param deviceName Name of the device.
   * @param properties Object with pairs: property name - property value
   */
  setProperty(
    deviceName: string,
    properties: { [key: string]: string | number | boolean },
  ): void;
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
 * CompiledModel represents a model that is compiled for a specific device
 * by applying multiple optimization transformations,
 * then mapping to compute kernels.
 */
interface CompiledModel {
  /** It gets all inputs of a compiled model. */
  inputs: Output[];
  /** It gets all outputs of a compiled model. */
  outputs: Output[];
  /**
   * It creates an inference request object used to infer the compiled model.
   * @return {InferRequest}
   */
  createInferRequest(): InferRequest;
  /**
   * It exports the compiled model to binary data.
   * @remarks
   * The exported model can be imported via the {@link Core.importModelSync}.
   * @return {Buffer} The binary data that contains the compiled model.
   */
  exportModelSync(): Buffer;
  /**
   * It gets a single output of a compiled model.
   * If a model has more than one output, this method throws an exception.
   * @returns {Output} A compiled model output.
   */
  output(): Output;
  /**
   * It gets output of a compiled model identified by an index.
   * @param index An output tensor index.
   * @returns {Output} A compiled model output.
   */
  output(index: number): Output;
  /**
   * It gets output of a compiled model identified by a tensorName.
   * @param name An output tensor name.
   * @returns {Output} A compiled model output.
   */
  output(name: string): Output;
  /**
   * It gets a single input of a compiled model.
   * If a model has more than one input, this method throws an exception.
   * @returns {Output} A compiled model input.
   */
  input(): Output;
  /**
   * It gets input of a compiled model identified by an index.
   * @param index An input tensor index.
   * @returns {Output} A compiled model input.
   */
  input(index: number): Output;
  /**
   * It gets input of a compiled model identified by a tensorName.
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
