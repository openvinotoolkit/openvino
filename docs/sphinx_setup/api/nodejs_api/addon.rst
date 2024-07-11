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

   interface NodeAddon {
       Core: CoreConstructor;
       PartialShape: PartialShapeConstructor;
       Tensor: TensorConstructor;
       element: typeof element;
       preprocess: {
           PrePostProcessor: PrePostProcessorConstructor;
           resizeAlgorithm: typeof resizeAlgorithm;
       };
   }

* **Defined in:**
  `addon.ts:591 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L192>`__


Properties
#####################


.. rubric:: Core

*

   .. code-block:: ts

      Core: CoreConstructor

   * **Type declaration:**

     - CoreConstructor: :doc:`CoreConstructor <./openvino-node/interfaces/CoreConstructor>`

   -  **Defined in:**
      `addon.ts:592 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L592>`__


.. rubric:: PartialShape

*

   .. code-block:: ts

      PartialShape: PartialShapeConstructor

   * **Type declaration:**

     - PartialShapeConstructor: :doc:`PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>`

   -  **Defined in:**
      `addon.ts:594 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L594>`__

.. rubric:: Tensor

*

   .. code-block:: ts

      Tensor: TensorConstructor

   * **Type declaration:**

     - TensorConstructor: :doc:`TensorConstructor <./openvino-node/interfaces/TensorConstructor>`

   -  **Defined in:**
      `addon.ts:593 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L593>`__


.. rubric:: element

*

   .. code-block:: ts

      element: typeof element

   * **Type declaration:**

     - element: typeof :doc:`element <./openvino-node/enums/element>`

   -  **Defined in:**
      `addon.ts:600 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L600>`__


.. rubric:: preprocess

*

   .. code-block:: ts

      preprocess: {
          PrePostProcessor: PrePostProcessorConstructor;
          resizeAlgorithm: typeof resizeAlgorithm;
      }

   * **Type declaration:**

     - PrePostProcessor: :doc:`PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>`
     - resizeAlgorithm: typeof :doc:`resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>`

   -  **Defined in:**
      `addon.ts:596 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L596>`__

