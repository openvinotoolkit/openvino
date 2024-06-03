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

/**
 * A user-defined model read by {@link Core.readModel}.
 */
interface Model {
  /**
   * It gets the friendly name for a model. If a friendly name is not set
   * via {@link Model.setFriendlyName}, a unique model name is returned.
   * @returns A string with a friendly name of the model.
   */
  getFriendlyName(): string;
  /**
   * It gets the unique name of the model.
   * @returns A string with the name of the model.
   */
  getName(): string;
  /**
   * It returns the shape of the element at the specified index.
   * @param index The index of the element.
   */
  getOutputShape(index: number): number[];
  /**
   * It returns the number of the model outputs.
   */
  getOutputSize(): number;
  /**
   * It gets the input of a model.
   * If a model has more than one input, this method throws an exception.
   */
  input(): Output;
  /**
   * It gets the input of a model identified by the tensor name.
   * @param name The tensor name.
   */
  input(name: string): Output;
  /**
   * It gets the input of a model identified by the index.
   * @param index The index of the input.
   */
  input(index: number): Output;
  /**
   * It returns true if any of the op’s defined in the model contains a partial
   * shape.
   */
  isDynamic(): boolean;
    /**
   * It gets the output of a model.
   * If a model has more than one output, this method throws an exception.
   */
  output(): Output;
  /**
   * It gets the output of a model identified by the tensor name.
   * @param name The tensor name.
   */
  output(name: string): Output;
  /**
   * It gets the output of a model identified by the index.
   * @param index The index of the input.
   */
  output(index: number): Output;
  /**
   * Sets a friendly name for a model. This does not overwrite the unique name
   * of the model and is retrieved via {@link Model.getFriendlyName}.
   * Used mainly for debugging.
   * @param name A string to set as the friendly name.
   */
  setFriendlyName(name: string): void;
  /**
   * It gets all inputs of a model as an array.
   */
  inputs: Output[];
  /**
   * It gets all outputs of a model as an array.
   */
  outputs: Output[];
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

/**
 * A {@link Tensor} is a lightweight class that represents data used for
 * inference. There are different ways to create a tensor. You can find them
 * in {@link TensorConstructor} section.
 */
interface Tensor {
  /**
   * This property provides access to the tensor's data.
   *
   * Its getter returns a subclass of a TypedArray that corresponds to the
   * tensor's element type e.g. Float32Array corresponds to float32. The
   * content of TypedArray subclass is a copy of the tensor's underlaying 
   * memory.
   *
   * Its setter fills the underlaying tensor’s memory by copying binary data
   * buffer from a TypedArray subclass. An exception will be thrown if the size
   * or type of array mismatches the tensor.
   */
  data: SupportedTypedArray;
  /**
   * It gets the tensor’s element type.
   */
  getElementType(): element;
  /**
   * It gets tensor's data.
   * @returns A subclass of a TypedArray that corresponds to the tensor's
   * element type e.g. Float32Array corresponds to float32.
   */
  getData(): SupportedTypedArray;
  /**
   * It gets the tensor’s shape.
   */
  getShape(): number[];
  /**
   * It gets the tensor’s size as a total number of elements.
   */
  getSize(): number;
}

/**
 * This interface contains constructors of {@link Tensor} class.
 *
 * @remarks
 * A tensor's memory is being shared with a TypedArray. That means
 * the responsibility of keeping the reference to TypedArray is on the side
 * of a user. Any action performed on the TypedArray will be reflected on this
 * tensor’s memory.
 */
interface TensorConstructor {
  /**
   * It constructs a tensor using element type and shape. New tensor data
   * will be allocated by default.
   * @param type The element type of a new tensor.
   * @param shape The shape of a new tensor.
   */
  new(type: element | elementTypeString, shape: number[]): Tensor;
  /**
   * It constructs a tensor using element type and shape. New tensor wraps
   * allocated host memory.
   * @param type The element type of a new tensor.
   * @param shape The shape of a new tensor.
   * @param tensorData A subclass of TypedArray that will be wrapped
   * by a {@link Tensor}.
   */
  new(type: element | elementTypeString, shape: number[],
    tensorData: SupportedTypedArray): Tensor;
}

/**
 * An {@link InferRequest} object is created using
 * {@link CompiledModel.createInferRequest} method and is specific for a given
 * deployed model. It is used to make predictions and it can be run in
 * asynchronous or synchronous manners.
 */
interface InferRequest {
  /**
   * It infers specified input(s) in the synchronous mode.
   * @remarks
   * Inputs have to be specified earlier using {@link InferRequest.setTensor}
   * or {@link InferRequest.setInputTensor}
   */
  infer(): { [outputName: string] : Tensor};
  /**
   * It infers specified input(s) in the synchronous mode.
   * @param inputData An object with the key-value pairs where the key is the
   * input name and value can be either a tensor or a TypedArray. TypedArray
   * will be wrapped into Tensor underneath using deployed model's input shape
   * and element type.
   */
  infer(inputData: { [inputName: string]: Tensor | SupportedTypedArray})
    : { [outputName: string] : Tensor};
  /**
   * It infers specified input(s) in the synchronous mode.
   * @param inputData An array with tensors or TypedArrays. TypedArrays will be
   * wrapped into Tensors underneath using the deployed model's input shape
   * and element type. Tensors and TypedArrays have to be passed in the correct
   * order if the model has multiple inputs.
   */
  infer(inputData: Tensor[] | SupportedTypedArray[])
    : { [outputName: string] : Tensor};
  /**
   * It infers specified input(s) in the asynchronous mode.
   * @param inputData An object with the key-value pairs where the key is the
   * input name and value is a tensor or an array with tensors. Tensors have to
   * be passed in the correct order if the model has multiple inputs.
   */
  inferAsync(inputData: { [inputName: string]: Tensor}
    | Tensor[] ): Promise<{ [outputName: string] : Tensor}>;
  /**
   * It gets the compiled model used by the InferRequest object.
   */
  getCompiledModel(): CompiledModel;
  /**
   * It gets an input tensor for inference.
   * @returns The input tensor for the model. If the model has several inputs,
   * an exception is thrown.
   */
  getInputTensor(): Tensor;
  /**
   * It gets an input tensor for inference.
   * @param idx An index of the tensor to get.
   * @returns A tensor at the specified index. If the tensor with the specified
   * idx is not found, an exception is thrown.
   */
  getInputTensor(idx: number): Tensor;
  /**
   * It gets an output tensor for inference.
   * @returns The output tensor for the model. If the model has several outputs,
   * an exception is thrown.
   */
  getOutputTensor(): Tensor;
 /**
   * It gets an output tensor for inference.
   * @param idx An index of the tensor to get.
   * @returns A tensor at the specified index. If the tensor with the specified
   * idx is not found, an exception is thrown.
   */
  getOutputTensor(idx?: number): Tensor;
  /**
   * It gets an input/output tensor for inference.
   *
   * @remarks
   * If the tensor with the specified name or port is not found, an exception
   * is thrown.
   * @param nameOrOutput A tensor name or output object.
   */
  getTensor(nameOrOutput: string | Output): Tensor;
  /**
   * It sets the input tensor to infer models with a single input.
   * @param tensor The input tensor. The element type and shape of the tensor
   * must match the model’s input element type and size. If the model has several
   * inputs, an exception is thrown.
   */
  setInputTensor(tensor: Tensor): void;
  /**
   * It sets the input tensor to infer.
   * @param idx The input tensor index. If idx is greater than the number of
   * model inputs, an exception is thrown.
   * @param tensor The input tensor. The element type and shape of the tensor
   * must match the model’s input element type and size.
   */
  setInputTensor(idx: number, tensor: Tensor): void;
  /**
   * It sets an output tensor to infer models with a single output.
   * @param tensor The output tensor. The element type and shape of the tensor
   * must match the model’s output element type and size. If the model has several
   * outputs, an exception is thrown.
   */
  setOutputTensor(tensor: Tensor): void;
  /**
   * It sets the output tensor to infer.
   * @param idx The output tensor index.
   * @param tensor The output tensor. The element type and shape of the tensor
   * must match the model’s output element type and size.
   */
  setOutputTensor(idx: number, tensor: Tensor): void;
  /**
   * It sets an input/output tensor to infer on.
   * @param name The input or output tensor name.
   * @param tensor The tensor. The element type and shape of the tensor
   * must match the model’s input/output element type and size.
   */
  setTensor(name: string, tensor: Tensor): void;
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
