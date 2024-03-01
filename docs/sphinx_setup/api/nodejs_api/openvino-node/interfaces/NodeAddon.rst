Property addon
===================

.. meta::
   :description: Explore the modules of openvino-node in Node.js API and their implementation
                 in Intel® Distribution of OpenVINO™ Toolkit.

The **openvino-node** package exports ``addon`` which contains the following properties:

.. code-block:: json

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
  `addon.ts:164 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L164>`__

Properties
#####################

.. rubric:: Core



.. code-block:: json

   Core: CoreConstructor

.. rubric:: Type declaration

- CoreConstructor: :doc:`CoreConstructor <CoreConstructor>`
-  Defined in
   `addon.ts:165 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L165>`__


.. rubric:: PartialShape



.. code-block:: json

   PartialShape: PartialShapeConstructor

.. rubric:: Type declaration

- PartialShapeConstructor: :doc:`PartialShapeConstructor <PartialShapeConstructor>`
-  Defined in
   `addon.ts:167 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L167>`__

.. rubric:: Tensor


.. code-block:: json

  Tensor: TensorConstructor

.. rubric:: Type declaration

- TensorConstructor: :doc:`TensorConstructor <TensorConstructor>`

-  Defined in
   `addon.ts:166 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L166>`__

.. rubric:: element



.. code-block:: json

   element: typeof element

.. rubric:: Type declaration

- element:typeof :doc:`element <../enums/element>`
-  Defined in
   `addon.ts:173 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L173>`__

.. rubric:: preprocess



.. code-block:: json

   preprocess: {
       PrePostProcessor: PrePostProcessorConstructor;
       resizeAlgorithm: typeof resizeAlgorithm;
   }


.. rubric:: Type declaration


- PrePostProcessor: :doc:`PrePostProcessorConstructor <PrePostProcessorConstructor>`
- resizeAlgorithm:typeof :doc:`resizeAlgorithm <../enums/resizeAlgorithm>`

-  Defined in
   `addon.ts:169 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L169>`__
