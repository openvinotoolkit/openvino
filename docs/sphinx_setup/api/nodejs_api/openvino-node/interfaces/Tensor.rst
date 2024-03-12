Interface Tensor
=====================

.. rubric:: Interface Tensor


.. code-block:: ts

   interface Tensor {
       data: number[];
       getData(): number[];
       getElementType(): element;
       getShape(): number[];
   }

- Defined in
  `addon.ts:60 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L60>`__

Properties
#####################

.. rubric:: data



.. code-block:: ts

   data: number[]

-  Defined in
   `addon.ts:61 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L61>`__



Methods
#####################

.. rubric:: getData


.. code-block:: ts

   getData(): number[]


**Returns** number[]


- Defined in
  `addon.ts:64 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L64>`__

.. rubric:: getElementType


.. code-block:: ts

   getElementType(): element


**Returns** :doc:`element <../enums/element>`

- Defined in
  `addon.ts:62 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L62>`__

.. rubric:: getShape

.. code-block:: ts

   getShape(): number[]


**Returns** number[]

- Defined in
  `addon.ts:63 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L63>`__
