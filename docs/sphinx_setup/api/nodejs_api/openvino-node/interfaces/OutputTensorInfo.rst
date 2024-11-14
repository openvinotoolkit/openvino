Interface OutputTensorInfo
==========================

.. code-block:: ts

   interface OutputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:581 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L581>`__


Methods
#####################


.. rubric:: setElementType

*

   .. code-block:: ts

      setElementType(elementType): InputTensorInfo

   * **Parameters:**

     - elementType: elementTypeString | element

   * **Returns** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:582 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L582>`__

.. rubric:: setLayout

*

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:583 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L583>`__

