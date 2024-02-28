Interface TensorConstructor
===========================

.. rubric:: Interface TensorConstructor


.. code-block:: json

   interface TensorConstructor {
       new TensorConstructornew (type, shape, tensorData?): Tensor;
   }

- Defined in
  `addon.ts:66 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L66>`__

.. rubric:: constructor



.. code-block:: json

   new TensorConstructor(type, shape, tensorData?): Tensor

**Parameters**

- type: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`
- shape: number[]
- ``Optional``

  .. code-block:: json

     tensorData: number[]|SupportedTypedArray


**Returns**  :doc:`Tensor <Tensor>`

- Defined in
  `addon.ts:67 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L67>`__
