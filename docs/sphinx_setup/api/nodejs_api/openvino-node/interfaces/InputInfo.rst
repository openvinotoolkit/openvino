Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

- Defined in
  `addon.ts:116 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L116>`__

Methods
#####################

.. rubric:: model



.. code-block:: ts

   model(): InputModelInfo

**Returns** :doc:`InputModelInfo <InputModelInfo>`

- Defined in
  `addon.ts:119 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L119>`__


.. rubric:: preprocess


.. code-block:: ts

   preprocess(): PreProcessSteps

**Returns** :doc:`PreProcessSteps <PreProcessSteps>`

- Defined in
  `addon.ts:118 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L118>`__


.. rubric:: tensor


.. code-block:: ts

   tensor(): InputTensorInfo


**Returns** :doc:`InputTensorInfo <InputTensorInfo>`

- Defined in
  `addon.ts:117 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L117>`__
