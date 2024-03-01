Interface InputTensorInfo
=========================


.. code-block:: json

   interface InputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
       setShape(shape): InputTensorInfo;
   }

- Defined in
  `addon.ts:98 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L98>`__

Methods
#####################

.. rubric:: setElementType



.. code-block:: json

   setElementType(elementType): InputTensorInfo

**Parameters**


- elementType: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

**Returns** :doc:`InputTensorInfo <InputTensorInfo>`



- Defined in
  `addon.ts:99 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L99>`__



.. rubric:: setLayout



.. code-block:: json

   setLayout(layout): InputTensorInfo

**Parameters**

- layout: string


**Returns** :doc:`InputTensorInfo <InputTensorInfo>`

- Defined in
  `addon.ts:100 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L100>`__

.. rubric:: setShape


.. code-block:: json

   setShape(shape): InputTensorInfo


**Parameters**

- shape: number[]

**Returns** :doc:`InputTensorInfo <InputTensorInfo>`

- Defined in
  `addon.ts:101 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L101>`__
