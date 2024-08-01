Interface OutputTensorInfo
==========================

.. code-block:: ts

   interface OutputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:530 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L530>`__


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
     `addon.ts:531 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L531>`__

.. rubric:: setLayout

*

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:532 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L532>`__

