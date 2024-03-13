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


The **openvino-node** package exports ``addon`` which contains the following properties:

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

- Defined in
  `addon.ts:164 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L164>`__

Properties
#####################

.. rubric:: Core


.. code-block:: ts

   Core: CoreConstructor

.. rubric:: Type declaration

- CoreConstructor: :doc:`CoreConstructor <./openvino-node/interfaces/CoreConstructor>`
-  Defined in
   `addon.ts:165 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L165>`__


.. rubric:: PartialShape



.. code-block:: ts

   PartialShape: PartialShapeConstructor

.. rubric:: Type declaration

- PartialShapeConstructor: :doc:`PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>`
-  Defined in
   `addon.ts:167 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L167>`__

.. rubric:: Tensor


.. code-block:: ts

  Tensor: TensorConstructor

.. rubric:: Type declaration

- TensorConstructor: :doc:`TensorConstructor <./openvino-node/interfaces/TensorConstructor>`

-  Defined in
   `addon.ts:166 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L166>`__

.. rubric:: element



.. code-block:: ts

   element: typeof element

.. rubric:: Type declaration

- element:typeof :doc:`element <./openvino-node/enums/element>`
-  Defined in
   `addon.ts:173 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L173>`__

.. rubric:: preprocess



.. code-block:: ts

   preprocess: {
       PrePostProcessor: PrePostProcessorConstructor;
       resizeAlgorithm: typeof resizeAlgorithm;
   }


.. rubric:: Type declaration


- PrePostProcessor: :doc:`PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>`
- resizeAlgorithm:typeof :doc:`resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>`

-  Defined in
   `addon.ts:169 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L169>`__
