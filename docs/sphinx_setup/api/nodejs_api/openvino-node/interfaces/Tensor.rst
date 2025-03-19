Interface Tensor
=====================

.. code-block:: ts

   interface Tensor {
       data: SupportedTypedArray;
       getElementType(): element;
       getData(): SupportedTypedArray;
       getShape(): number[];
       getSize(): number;
       isContinuous(): boolean;

   }

The ``Tensor`` is a lightweight class that represents data used for
inference. There are different ways to create a tensor. You can find them
in :doc:`TensorConstructor <TensorConstructor>` section.

* **Defined in:**
  `addon.ts:390 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L390>`__


Properties
#####################


.. rubric:: data

*

   .. code-block:: ts

      data: SupportedTypedArray

   This property provides access to the tensor's data.

   Its getter returns a subclass of TypedArray that corresponds to the
   tensor element type, e.g. ``Float32Array`` corresponds to ``float32``. The
   content of the ``TypedArray`` subclass is a copy of the tensor underlaying
   memory.

   Its setter fills the underlaying tensor memory by copying the binary data
   buffer from the ``TypedArray`` subclass. An exception will be thrown if the size
   or type of array does not match the tensor.

   -  **Defined in:**
      `addon.ts:403 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L403>`__


Methods
#####################


.. rubric:: getData

*

   .. code-block:: ts

      getData(): SupportedTypedArray;

   It gets tensor data.

   * **Returns:** SupportedTypedArray

     A subclass of ``TypedArray`` corresponding to the tensor
     element type, e.g. ``Float32Array`` corresponds to float32.

   * **Defined in:**
     `addon.ts:413 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L413>`__

.. rubric:: getElementType

*

   .. code-block:: ts

      getElementType(): element

   It gets the tensor element type.

   * **Returns:** :doc:`element <../enums/element>`

   * **Defined in:**
     `addon.ts:407 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L407>`__


.. rubric:: getShape

*

   .. code-block:: ts

      getShape(): number[]

   It gets the tensor shape.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:417 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L417>`__


.. rubric:: getSize

*

   .. code-block:: ts

      getSize(): number[]

   It gets the tensor size as a total number of elements.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:421 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L421>`__


.. rubric:: isContinuous

*

   .. code-block:: ts

      isContinuous(): boolean;

   Reports whether the tensor is continuous or not.

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:425 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L425>`__
  