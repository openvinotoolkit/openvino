Interface Tensor
=====================

.. rubric:: Interface Tensor


.. code-block:: json

   interface Tensor {
       data: number[];
       getData(): number[];
       getElementType(): element;
       getShape(): number[];
   }

- Defined in
  `addon.ts:60 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L60>`__

Properties
#####################

.. rubric:: data



.. code-block:: json

   data: number[]

-  Defined in
   `addon.ts:61 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L61>`__



Methods
#####################

.. rubric:: getData


.. code-block:: json

   getData():number[]


**Returns** number[]


- Defined in
  `addon.ts:64 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L64>`__

.. rubric:: getElementType


.. code-block:: json

   getElementType():element


**Returns** :doc:`element <../enums/element>`

- Defined in
  `addon.ts:62 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L62>`__

.. rubric:: getShape

.. code-block:: json

   getShape():number[]


**Returns** number[]

- Defined in
  `addon.ts:63 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L63>`__
