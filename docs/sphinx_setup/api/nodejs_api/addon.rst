Property addon
===================

.. meta::
   :description: Explore the modules of openvino-node in Node.js API and their implementation
                 in Intel® Distribution of OpenVINO™ Toolkit.

.. toctree::
   :maxdepth: 3
   :hidden:

   element <./openvino-node/enums/element>
   resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>
   CompiledModel <./openvino-node/interfaces/CompiledModel>
   Core <./openvino-node/interfaces/Core>
   CoreConstructor <./openvino-node/interfaces/CoreConstructor>
   InferRequest <./openvino-node/interfaces/InferRequest>
   InputInfo <./openvino-node/interfaces/InputInfo>
   InputModelInfo <./openvino-node/interfaces/InputModelInfo>
   InputTensorInfo <./openvino-node/interfaces/InputTensorInfo>
   Model <./openvino-node/interfaces/Model>
   Output <./openvino-node/interfaces/Output>
   OutputInfo <./openvino-node/interfaces/OutputInfo>
   OutputTensorInfo <./openvino-node/interfaces/OutputTensorInfo>
   OVAny <./openvino-node/types/OVAny>
   PartialShape <./openvino-node/interfaces/PartialShape>
   PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>
   PrePostProcessor <./openvino-node/interfaces/PrePostProcessor>
   PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>
   PreProcessSteps <./openvino-node/interfaces/PreProcessSteps>
   Tensor <./openvino-node/interfaces/Tensor>
   TensorConstructor <./openvino-node/interfaces/TensorConstructor>
   Dimension <./openvino-node/types/Dimension>
   elementTypeString <./openvino-node/types/elementTypeString>
   SupportedTypedArray <./openvino-node/types/SupportedTypedArray>

The **openvino-node** package exports ``addon`` which contains the following properties:

.. rubric:: Interface NodeAddon

.. code-block:: ts

   export interface NodeAddon {
       Core: CoreConstructor;
       Tensor: TensorConstructor;
       PartialShape: PartialShapeConstructor;

       preprocess: {
         resizeAlgorithm: typeof resizeAlgorithm;
         PrePostProcessor: PrePostProcessorConstructor;
       };
       saveModelSync(model: Model, path: string, compressToFp16?: boolean): void;
       element: typeof element;
     }

* **Defined in:**
  `addon.ts:669 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L669>`__


Properties
#####################


.. rubric:: Core

*

   .. code-block:: ts

      Core: CoreConstructor

   * **Type declaration:**

     - CoreConstructor: :doc:`CoreConstructor <./openvino-node/interfaces/CoreConstructor>`

   -  **Defined in:**
      `addon.ts:670 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L670>`__


.. rubric:: PartialShape

*

   .. code-block:: ts

      PartialShape: PartialShapeConstructor

   * **Type declaration:**

     - PartialShapeConstructor: :doc:`PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>`

   -  **Defined in:**
      `addon.ts:672 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L672>`__

.. rubric:: Tensor

*

   .. code-block:: ts

      Tensor: TensorConstructor

   * **Type declaration:**

     - TensorConstructor: :doc:`TensorConstructor <./openvino-node/interfaces/TensorConstructor>`

   -  **Defined in:**
      `addon.ts:671 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L671>`__


.. rubric:: element

*

   .. code-block:: ts

      element: typeof element

   * **Type declaration:**

     - element: typeof :doc:`element <./openvino-node/enums/element>`

   -  **Defined in:**
      `addon.ts:678 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L678>`__


.. rubric:: preprocess

*

   .. code-block:: ts

      preprocess: {
          resizeAlgorithm: typeof resizeAlgorithm;
          PrePostProcessor: PrePostProcessorConstructor;
      }

   * **Type declaration:**

     - resizeAlgorithm: typeof :doc:`resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>`
     - PrePostProcessor: :doc:`PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>`

   -  **Defined in:**
      `addon.ts:674 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L674>`__


.. rubric:: saveModelSync

*

   .. code-block:: ts

      saveModelSync(model: Model, path: string, compressToFp16?: boolean): void;


   This method saves a model to IR (xml and bin files), applying all
   necessary transformations that are usually added during model conversion.
   Particularly, weights are compressed to FP16 by default, and debug information
   in model nodes is cleaned up.

   * **Parameters:**

     - model: :doc:`Model <openvino-node/interfaces/Model>`

       A model which will be converted to IR and saved.

     - path: string

       A path for saving the model.

     - ``Optional``

       - compressToFp16: boolean

         Compression of weights to FP16 floating point precision. The default value is `true` .

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:692 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L692>`__

