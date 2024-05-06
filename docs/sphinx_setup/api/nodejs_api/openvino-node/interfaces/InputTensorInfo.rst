Interface InputTensorInfo
=========================

.. code-block:: ts

   interface InputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
       setShape(shape): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:126 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L126>`__


Methods
#####################


.. rubric:: setElementType

.. container:: m-4

   .. code-block:: ts

      setElementType(elementType): InputTensorInfo

   * **Parameters:**

     - elementType: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:127 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L127>`__


.. rubric:: setLayout

.. container:: m-4

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:128 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L128>`__


.. rubric:: setShape

.. container:: m-4

   .. code-block:: ts

      setShape(shape): InputTensorInfo

   * **Parameters:**

     - shape: number[]

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:129 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L129>`__

