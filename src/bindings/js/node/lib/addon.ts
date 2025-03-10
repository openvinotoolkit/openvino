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

type OVAny = string | number | boolean;

/**
 * Core represents an OpenVINO runtime Core entity.
 *
 * User applications can create several Core class instances,
 * but in this case, the underlying plugins
 * are created multiple times and not shared between several Core instances.
 * It is recommended to have a single Core instance per application.
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
   * You can create as many compiled models as needed and use them
   * simultaneously (up to the limitation of the hardware resources).
   * @param model The {@link Model} object acquired from {@link Core.readModel}
   * @param deviceName The name of a device, to which the model is loaded.
   * @param config An object with the key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  compileModel(
    model: Model,
    deviceName: string,
    config?: Record<string, OVAny>,
  ): Promise<CompiledModel>;
  /**
   * Asynchronously reads a model and creates a compiled model
   * from the IR/ONNX/PDPD file.
   *
   * This can be more efficient
   * than using {@link Core.readModel} + core.compileModel(Model) flow
   * especially for cases when caching is enabled and a cached model is
   * available. You can create as many compiled models as needed and use
   * them simultaneously (up to the limitation of the hardware resources).
   * @param modelPath The path to a model.
   * @param deviceName The name of a device, to which a model is loaded.
   * @param config An object with the key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  compileModel(
    modelPath: string,
    deviceName: string,
    config?: Record<string, OVAny>,
  ): Promise<CompiledModel>;
  /**
   * A synchronous version of {@link Core.compileModel}.
   * It creates a compiled model from a source model object.
   */
  compileModelSync(
    model: Model,
    deviceName: string,
    config?: Record<string, OVAny>,
  ): CompiledModel;
  /**
   * A synchronous version of {@link Core.compileModel}.
   * It reads a model and creates a compiled model from the IR/ONNX/PDPD file.
   */
  compileModelSync(
    modelPath: string,
    deviceName: string,
    config?: Record<string, OVAny>,
  ): CompiledModel;
  /**
   * It returns a list of available inference devices.
   * Core objects go over all registered plugins.
   * @returns The list of devices may include any of the following: CPU, GPU.0,
   * GPU.1, NPU… If there is more than one device of a specific type, they are
   * enumerated with .# suffix. Such enumerated devices can later be used
   * as a device name in all Core methods, like compile_model, query_model,
   * set_property and so on.
   */
  getAvailableDevices(): string[];
  /**
   * It gets the properties dedicated to device behaviour.
   * @param propertyName A property name.
   */
  getProperty(propertyName: string): OVAny;

  /**
   * It gets the properties dedicated to device behaviour.
   * @param deviceName The name of a device, the properties of which you get.
   * @param propertyName Property name.
   */
  getProperty(
    deviceName: string,
    propertyName: string,
  ): OVAny;
  /**
   * It returns information on the version of device plugins.
   * @param deviceName A device name to identify a plugin.
   */
  getVersions(deviceName: string): {
    [deviceName: string]: {
      buildNumber: string;
      description: string;
    };
  };
  /**
   * Asynchronously imports a previously exported compiled model.
   * @param modelStream The input stream that contains a model,
   * previously exported with the {@link CompiledModel.exportModelSync} method.
   * @param device The name of a device, for which you import a compiled model.
   * Note, if the device name was not used to compile the original model,
   * an exception is thrown.
   * @param config An object with the key-value pairs
   * (property name, property value): relevant only for this load operation.
   */
  importModel(
    modelStream: Buffer,
    device: string,
    config?: Record<string, OVAny>,
  ): Promise<CompiledModel>;
  /**
   * A synchronous version of {@link Core.importModel}.
   * It imports a previously exported compiled model.
   */
  importModelSync(
    modelStream: Buffer,
    device: string,
    config?: Record<string, OVAny>,
  ): CompiledModel;
  /**
   * It reads models from the IR / ONNX / PDPD / TF and TFLite formats.
   * @param modelPath The path to a model
   * in the IR / ONNX / PDPD / TF or TFLite format.
   * @param weightsPath The path to a data file for the IR format (.bin):
   * if the path is empty, it tries to read the bin file with the same name
   * as xml and if the bin file with the same name was not found, it loads
   * IR without weights.
   * For the ONNX format (.onnx), the weights parameter is not used.
   * For the PDPD format (.pdmodel), the weights parameter is not used.
   * For the TF format (.pb), the weights parameter is not used.
   * For the TFLite format (*.tflite), the weights parameter is not used.
   */
  readModel(modelPath: string, weightsPath?: string): Promise<Model>;
  /**
   * It reads models from IR / ONNX / PDPD / TF and TFLite formats.
   * @param model A string with model in IR / ONNX / PDPD / TF
   * and TFLite format.
   * @param weights Tensor with weights. Reading ONNX / PDPD / TF
   * and TFLite models doesn’t support loading weights from weights tensors.
   */
  readModel(model: string, weights: Tensor): Promise<Model>;
  /**
   * It reads models from the IR / ONNX / PDPD / TF and TFLite formats.
   * @param modelBuffer Binary data with a model
   * in the IR / ONNX / PDPD / TF or TFLite format.
   * @param weightsBuffer Binary data with tensor data.
   */
  readModel(
    modelBuffer: Uint8Array,
    weightsBuffer?: Uint8Array,
  ): Promise<Model>;
  /**
   * A synchronous version of {@link Core.readModel}.
   * It reads models from the IR / ONNX / PDPD / TF and TFLite formats.
   */
  readModelSync(modelPath: string, weightsPath?: string): Model;
  /**
   * A synchronous version of {@link Core.readModel}.
   * It reads models from the IR / ONNX / PDPD / TF and TFLite formats.
   */
  readModelSync(model: string, weights: Tensor): Model;
  /**
   * A synchronous version of {@link Core.readModel}.
   * It reads models from the IR / ONNX / PDPD / TF and TFLite formats.
   */
  readModelSync(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Model;
  /**
   * It sets the properties.
   * @param properties An object with the property name - property value pairs.
   */
  setProperty(properties: Record<string, OVAny>): void;
  /**
   * It sets the properties for a device.
   * @param deviceName The name of a device.
   * @param properties An object with the property name - property value pairs.
   */
  setProperty(deviceName: string, properties: Record<string, OVAny>): void;
  /**
   * It queries the device if it supports specified model with the specified
   * properties.
   * @param model The passed model to query the property.
   * @param deviceName The name of a device.
   * @param properties An object with the property name - property value pairs.
   * An object with the key-value pairs. (property name, property value).
   */
  queryModel(
    model: Model,
    deviceName: string,
    properties?: Record<string, OVAny>,
  ): { [key: string]: string };
}
interface CoreConstructor {
  new (): Core;
}

/**
 * A user-defined model read by {@link Core.readModel}.
 */
interface Model {
  /**
   * It returns a cloned model.
   */
  clone(): Model;
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
   * It gets the element type of a specific output of the model.
   * @param index The index of the output.
   */
  getOutputElementType(index: number): string;
  /**
   * It gets the input of the model.
   * If a model has more than one input, this method throws an exception.
   */
  input(): Output;
  /**
   * It gets the input of the model identified by the tensor name.
   * @param name The tensor name.
   */
  input(name: string): Output;
  /**
   * It gets the input of the model identified by the index.
   * @param index The index of the input.
   */
  input(index: number): Output;
  /**
   * It returns true if any of the op’s defined in the model contains a partial
   * shape.
   */
  isDynamic(): boolean;
  /**
   * It gets the output of the model.
   * If a model has more than one output, this method throws an exception.
   */
  output(): Output;
  /**
   * It gets the output of the model identified by the tensor name.
   * @param name The tensor name.
   */
  output(name: string): Output;
  /**
   * It gets the output of the model identified by the index.
   * @param index The index of the input.
   */
  output(index: number): Output;
  /**
   * Sets a friendly name for the model. This does not overwrite the unique
   * model name and is retrieved via {@link Model.getFriendlyName}.
   * Mainly used for debugging.
   * @param name The string to set as the friendly name.
   */
  setFriendlyName(name: string): void;
  /**
   * It gets all the model inputs as an array.
   */
  inputs: Output[];
  /**
   * It gets all the model outputs as an array
   */
  outputs: Output[];
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
   * It gets the property for the current compiled model.
   * @param propertyName A string to get the property value.
   * @returns The property value.
   */
  getProperty(propertyName: string): OVAny;
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
  /**
   * It sets properties for the current compiled model. Properties
   * can be retrieved via {@link CompiledModel.getProperty}.
   * @param property An object with the key-value pairs.
   * (property name, property value)
   */
  setProperty(properties: Record<string, OVAny>): void;
}

/**
 * The {@link Tensor} is a lightweight class that represents data used for
 * inference. There are different ways to create a tensor. You can find them
 * in {@link TensorConstructor} section.
 */
interface Tensor {
  /**
   * This property provides access to the tensor's data.
   *
   * Its getter returns a subclass of TypedArray that corresponds to the
   * tensor element type, e.g. Float32Array corresponds to float32. The
   * content of the TypedArray subclass is a copy of the tensor underlaying
   * memory.
   *
   * Its setter fills the underlaying tensor memory by copying the binary data
   * buffer from the TypedArray subclass. An exception will be thrown if the
   * size or type of array does not match the tensor.
   */
  data: SupportedTypedArray;
  /**
   * It gets the tensor element type.
   */
  getElementType(): element;
  /**
   * It gets tensor data.
   * @returns A subclass of TypedArray corresponding to the tensor
   * element type, e.g. Float32Array corresponds to float32.
   */
  getData(): SupportedTypedArray;
  /**
   * It gets the tensor shape.
   */
  getShape(): number[];
  /**
   * It gets the tensor size as a total number of elements.
   */
  getSize(): number;
  /**
   * Reports whether the tensor is continuous or not.
   */
  isContinuous(): boolean;
}

/**
 * This interface contains constructors of the {@link Tensor} class.
 *
 * @remarks
 * The tensor memory is shared with the TypedArray. That is,
 * the responsibility for maintaining the reference to the TypedArray lies with
 * the user. Any action performed on the TypedArray will be reflected in this
 * tensor memory.
 */
interface TensorConstructor {
  /**
   * It constructs a tensor using the element type and shape. The new tensor
   * data will be allocated by default.
   * @param type The element type of the new tensor.
   * @param shape The shape of the new tensor.
   */
  new (type: element | elementTypeString, shape: number[]): Tensor;
  /**
   * It constructs a tensor using the element type and shape. The new tensor
   * wraps allocated host memory.
   * @param type The element type of the new tensor.
   * @param shape The shape of the new tensor.
   * @param tensorData A subclass of TypedArray that will be wrapped
   * by a {@link Tensor}.
   */
  new (
    type: element | elementTypeString,
    shape: number[],
    tensorData: SupportedTypedArray,
  ): Tensor;
  /**
   * It constructs a tensor using the element type and shape. The strings from
   * the array are used to fill the new tensor. Each element of a string tensor
   * is a string of arbitrary length, including an empty string.
   */
  new (tensorData: string[]): Tensor;
}

/**
 * The {@link InferRequest} object is created using
 * {@link CompiledModel.createInferRequest} method and is specific for a given
 * deployed model. It is used to make predictions and can be run in
 * asynchronous or synchronous manners.
 */
interface InferRequest {
  /**
   * It infers specified input(s) in the synchronous mode.
   * @remarks
   * Inputs have to be specified earlier using {@link InferRequest.setTensor}
   * or {@link InferRequest.setInputTensor}
   */
  infer(): { [outputName: string]: Tensor };
  /**
   * It infers specified input(s) in the synchronous mode.
   * @param inputData An object with the key-value pairs where the key is the
   * input name and value can be either a tensor or a TypedArray.
   * TypedArray will be wrapped into Tensor underneath using the input shape
   * and element type of the deployed model.
   */
  infer(inputData: { [inputName: string]: Tensor | SupportedTypedArray }): {
    [outputName: string]: Tensor;
  };
  /**
   * It infers specified input(s) in the synchronous mode.
   * @param inputData An array with tensors or TypedArrays. TypedArrays will be
   * wrapped into Tensors underneath using the input shape and element type
   * of the deployed model. If the model has multiple inputs, the Tensors
   * and TypedArrays must be passed in the correct order.
   */
  infer(inputData: Tensor[] | SupportedTypedArray[]): {
    [outputName: string]: Tensor;
  };
  /**
   * It infers specified input(s) in the asynchronous mode.
   * @param inputData An object with the key-value pairs where the key is the
   * input name and value is a tensor or an array with tensors. If the model has
   * multiple inputs, the Tensors must be passed in the correct order.
   */
  inferAsync(
    inputData: { [inputName: string]: Tensor } | Tensor[],
  ): Promise<{ [outputName: string]: Tensor }>;
  /**
   * It gets the compiled model used by the InferRequest object.
   */
  getCompiledModel(): CompiledModel;
  /**
   * It gets the input tensor for inference.
   * @returns The input tensor for the model. If the model has several inputs,
   * an exception is thrown.
   */
  getInputTensor(): Tensor;
  /**
   * It gets the input tensor for inference.
   * @param idx An index of the tensor to get.
   * @returns A tensor at the specified index. If the tensor with the specified
   * idx is not found, an exception is thrown.
   */
  getInputTensor(idx: number): Tensor;
  /**
   * It gets the output tensor for inference.
   * @returns The output tensor for the model. If the model has several outputs,
   * an exception is thrown.
   */
  getOutputTensor(): Tensor;
  /**
   * It gets the output tensor for inference.
   * @param idx An index of the tensor to get.
   * @returns A tensor at the specified index. If the tensor with the specified
   * idx is not found, an exception is thrown.
   */
  getOutputTensor(idx?: number): Tensor;
  /**
   * It gets an input/output tensor for inference.
   *
   * @remarks
   * If a tensor with the specified name or port is not found, an exception
   * is thrown.
   * @param nameOrOutput The name of the tensor or output object.
   */
  getTensor(nameOrOutput: string | Output): Tensor;
  /**
   * It sets the input tensor to infer models with a single input.
   * @param tensor The input tensor. The element type and shape of the tensor
   * must match the type and size of the model's input element. If the model
   * has several inputs, an exception is thrown.
   */
  setInputTensor(tensor: Tensor): void;
  /**
   * It sets the input tensor to infer.
   * @param idx The input tensor index. If idx is greater than the number of
   * model inputs, an exception is thrown.
   * @param tensor The input tensor. The element type and shape of the tensor
   * must match the input element type and size of the model.
   */
  setInputTensor(idx: number, tensor: Tensor): void;
  /**
   * It sets the output tensor to infer models with a single output.
   * @param tensor The output tensor. The element type and shape of the tensor
   * must match the output element type and size of the model. If the model
   * has several outputs, an exception is thrown.
   */
  setOutputTensor(tensor: Tensor): void;
  /**
   * It sets the output tensor to infer.
   * @param idx The output tensor index.
   * @param tensor The output tensor. The element type and shape of the tensor
   * must match the output element type and size of the model.
   */
  setOutputTensor(idx: number, tensor: Tensor): void;
  /**
   * It sets the input/output tensor to infer.
   * @param name The input or output tensor name.
   * @param tensor The tensor. The element type and shape of the tensor
   * must match the input/output element type and size of the model.
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
  setElementType(elementType: element | elementTypeString): InputTensorInfo;
  setLayout(layout: string): InputTensorInfo;
  setShape(shape: number[]): InputTensorInfo;
}

interface OutputTensorInfo {
  setElementType(elementType: element | elementTypeString): InputTensorInfo;
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
  new (model: Model): PrePostProcessor;
}

interface PartialShape {
  isStatic(): boolean;
  isDynamic(): boolean;
  toString(): string;
  getDimensions(): Dimension[];
}

/**
 * This interface contains constructor of the {@link PartialShape} class.
 */
interface PartialShapeConstructor {
  /**
   * It constructs a PartialShape by passed string.
   * Omit parameter to create empty shape.
   * @param [shape] String representation of the shape.
   */
  new (shape?: string): PartialShape;
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
  Core: CoreConstructor;
  Tensor: TensorConstructor;
  PartialShape: PartialShapeConstructor;

  preprocess: {
    resizeAlgorithm: typeof resizeAlgorithm;
    PrePostProcessor: PrePostProcessorConstructor;
  };

  /**
   * It saves a model into IR files (xml and bin).
   * Floating point weights are compressed to FP16 by default.
   * This method saves a model to IR applying all necessary transformations
   * that usually applied in model conversion flow provided by mo tool.
   * Particularly, floating point weights are compressed to FP16,
   * debug information in model nodes are cleaned up, etc.
   * @param model The model which will be
   * converted to IR representation and saved.
   * @param path The path for saving the model.
   * @param compressToFp16 Whether to compress
   * floating point weights to FP16. Default is set to `true`.
   */
  saveModelSync(model: Model, path: string, compressToFp16?: boolean): void;

  element: typeof element;
}

export default // eslint-disable-next-line @typescript-eslint/no-var-requires
require('../bin/ov_node_addon.node') as NodeAddon;
