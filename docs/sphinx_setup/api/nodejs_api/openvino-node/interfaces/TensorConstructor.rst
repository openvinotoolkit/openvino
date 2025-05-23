Interface TensorConstructor
===========================

.. code-block:: ts

   interface TensorConstructor {
       new TensorConstructor (type, shape): Tensor;
       new TensorConstructor (type, shape, tensorData): Tensor;
   }

This interface contains constructors of the :doc:`Tensor <Tensor>` class.

The tensor memory is shared with the ``TypedArray``. That is,
the responsibility for maintaining the reference to the ``TypedArray`` lies with
the user. Any action performed on the ``TypedArray`` will be reflected in this
tensor memory.

* **Defined in:**
  `addon.ts:433 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L433>`__


Constructors
#####################


.. rubric:: constructor

*

   .. code-block:: ts

      new TensorConstructor(type, shape): Tensor

   It constructs a tensor using the element type and shape. The new tensor data
   will be allocated by default.

   * **Parameters:**

     - type: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

       The element type of the new tensor.

     - shape: number[]

       The shape of the new tensor.

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:440 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L440>`__


   .. code-block:: ts

       new TensorConstructor(type, shape, tensorData): Tensor

   It constructs a tensor using the element type and shape. The new tensor wraps
   allocated host memory.

   * **Parameters:**

     - type: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

       The element type of the new tensor.

     - shape: number[]

       The shape of the new tensor.

     - tensorData: SupportedTypedArray

       A subclass of TypedArray that will be wrapped by a :doc:`Tensor <Tensor>`

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:449 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L449>`__


   .. code-block:: ts

       new TensorConstructor(tensorData: string[]): Tensor;

   It constructs a string tensor. The strings from
   the array are used to fill tensor data. Each element of a string tensor
   is a string of arbitrary length.

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:459 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L459>`__

