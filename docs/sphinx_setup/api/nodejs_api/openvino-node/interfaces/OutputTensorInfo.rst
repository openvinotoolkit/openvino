Interface OutputTensorInfo
==========================

.. code-block:: json

   interface OutputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
   }

- Defined in
  `addon.ts:104 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L104>`__

Methods
#####################

.. rubric:: setElementType


.. code-block:: json

   setElementType(elementType): InputTensorInfo

**Parameters**

- elementType: elementTypeString | element

**Returns** :doc:`InputTensorInfo <InputTensorInfo>`

- Defined in
  `addon.ts:105 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L105>`__

.. rubric:: setLayout


.. code-block:: json

   setLayout(layout): InputTensorInfo


**Parameters**

- layout: string

**Returns** :doc:`InputTensorInfo <InputTensorInfo>`

- Defined in
  `addon.ts:106 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L106>`__
