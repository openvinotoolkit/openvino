OpenVINO Node.js API
=====================

.. meta::
   :description: Explore Node.js API and implementation of its features in Intel®
                 Distribution of OpenVINO™ Toolkit.


OpenVINO Node.js API is distributed as an *openvino-node* npm package that contains JavaScript
wrappers with TypeScript types descriptions and a script that downloads the OpenVINO Node.js
bindings for current OS.⠀

Use openvino-node package
#########################

1. Import openvino-node package. Use the ``addon`` property to reach general exposed entities:

   .. code-block:: js

      const { addon: ov } = require('openvino-node');


2. Load and compile a model, then prepare a tensor with input data. Finally, run inference
   on the model with it to get the model output tensor:

   .. code-block:: js

      const { addon: ov } = require('openvino-node');
      // Load model
      const core = new ov.Core();
      const model = await ov.readModel('path/to/model', 'path/to/model/weights');
      // Compile model
      const compiledModel = await ov.compileModel(model, 'CPU');
      // Prepare tensor with input data
      const tensorData = new Float32Array(image.data);
      const shape = [1, image.rows, image.cols, 3];
      const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);
      const inferRequest = compiledModel.createInferRequest();
      const modelOutput = inferRequest.infer([inputTensor]);


For more extensive examples of use, refer to the following scripts:

- `Hello Classification Sample <https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/hello_classification/hello_classification.js>`__
- `Hello Reshape SSD Sample <https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/hello_reshape_ssd/hello_reshape_ssd.js>`__
- `Image Classification Async Sample <https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/classification_sample_async/classification_sample_async.js>`__

OpenVINO API features
#####################

.. list-table::
   :widths: 15 85
   :class: nodejs-features

   * - ``addon``
     -
       .. code-block:: ts

          Core()
          Tensor()
          PartialShape()
          element
          preprocess:
            resizeAlgorithms
            PrePostProcessor()

   * - ``CompiledModel``
     -
       .. code-block:: ts

          outputs: Output[]
          inputs: Output[]
          constructor()
          output(nameOrId?: string | number): Output
          input(nameOrId?: string | number): Output
          createInferRequest(): InferRequest

   * - ``Core``
     -
       .. code-block:: ts

          constructor()
          compileModel(model: Model, device: string, config?: { [option: string]: string }): Promise<CompiledModel>
          compileModelSync(model: Model, device: string, config?: { [option: string]: string }): CompiledModel
          readModel(modelPath: string, binPath?: string): Promise<Model>
          readModel(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Promise<Model>;
          readModelSync(modelPath: string, binPath?: string): Model
          readModelSync(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Model;

   * - ``InferRequest``
     -
       .. code-block:: ts

          constructor()
          setTensor(name: string, tensor: Tensor): void
          setInputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void
          setOutputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void
          getTensor(nameOrOutput: string | Output): Tensor
          getInputTensor(idx?: number): Tensor
          getOutputTensor(idx?: number): Tensor
          getCompiledModel(): CompiledModel
          inferAsync(inputData?: { [inputName: string]: Tensor |SupportedTypedArray} | Tensor[] | SupportedTypedArray[]): Promise<{ [outputName: string] : Tensor}>;
          infer(inputData?: { [inputName: string]: Tensor |SupportedTypedArray} | Tensor[] | SupportedTypedArray[]): { [outputName: string] : Tensor};

   * - ``InputInfo``
     -
       .. code-block:: ts

          tensor(): InputTensorInfo;
          preprocess(): PreProcessSteps;
          model(): InputModelInfo;

   * - ``InputModelInfo``
     -
       .. code-block:: ts

          setLayout(layout: string): InputModelInfo;

   * - ``InputTensorInfo``
     -
       .. code-block:: ts

          setElementType(elementType: element | elementTypeString ): InputTensorInfo;
          setLayout(layout: string): InputTensorInfo;
          setShape(shape: number[]): InputTensorInfo;

   * - ``Model``
     -
       .. code-block:: ts

          outputs: Output[]
          inputs: Output[]
          output(nameOrId?: string | number): Output
          input(nameOrId?: string | number): Output
          getName(): string

   * - ``Output``
     -
       .. code-block:: ts

          anyName: string;
          shape: number[];

          constructor()
          toString(): string
          getAnyName(): string
          getShape(): number[]
          getPartialShape(): number[]

   * - ``OutputInfo``
     -
       .. code-block:: ts

          tensor(): OutputTensorInfo;

   * - ``OutputTensorInfo``
     -
       .. code-block:: ts

          setElementType(elementType: element | elementTypeString ): InputTensorInfo;
          setLayout(layout: string): InputTensorInfo;

   * - ``PrePostProcessor``
     -
       .. code-block:: ts

          constructor(model: Model)
          build(): PrePostProcessor
          input(): InputInfo
          output(): OutputInfo

   * - ``preprocess.element``
     - u8, u16, u32, i8, i16, i32, i64, f32, f64

   * - ``preprocess.resizeAlgorithm``
     - RESIZE_CUBIC, RESIZE_LINEAR

   * - ``PreProcessSteps``
     -
       .. code-block:: ts

          resize(algorithm: resizeAlgorithm | string): PreProcessSteps;

   * - ``Tensor``
     -
       .. code-block:: ts

          data: number[]
          constructor(type: element, shape: number[], tensorData?: number[] | SupportedTypedArray): Tensor
          getElementType(): element
          getShape(): number[]
          getData(): number[]

