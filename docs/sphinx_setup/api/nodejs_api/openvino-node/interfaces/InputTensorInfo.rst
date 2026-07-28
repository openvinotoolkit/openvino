Interface InputTensorInfo
=========================

.. code-block:: ts

   interface InputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
       setShape(shape): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:593 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L593>`__


Methods
#####################


.. rubric:: setElementType

*

   .. code-block:: ts

      setElementType(elementType): InputTensorInfo

   * **Parameters:**

     - elementType: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:594 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L594>`__


.. rubric:: setLayout

*

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:595 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L595>`__


.. rubric:: setShape

*

   .. code-block:: ts

      setShape(shape): InputTensorInfo

   * **Parameters:**

     - shape: number[]

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:596 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L596>`__

