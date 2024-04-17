Interface TensorConstructor
===========================

.. rubric:: Interface TensorConstructor


.. code-block:: ts

   interface TensorConstructor {
       new Tensor(type, shape, tensorData?): Tensor;
   }

- Defined in
  `addon.ts:66 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L66>`__

.. rubric:: constructor



.. code-block:: ts

   new Tensor(type, shape, tensorData?): Tensor

**Parameters**

- type: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`
- shape: number[]
- ``Optional``

  .. code-block:: ts

     tensorData: number[]|SupportedTypedArray


**Returns**  :doc:`Tensor <Tensor>`

- Defined in
  `addon.ts:67 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L67>`__
