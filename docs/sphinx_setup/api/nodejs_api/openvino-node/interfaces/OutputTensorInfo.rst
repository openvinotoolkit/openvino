Interface OutputTensorInfo
==========================

.. code-block:: ts

   interface OutputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:132 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L132>`__


Methods
#####################


.. rubric:: setElementType

.. container:: m-4

   .. code-block:: ts

      setElementType(elementType): InputTensorInfo

   * **Parameters:**

     - elementType: elementTypeString | element

   * **Returns** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:133 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L133>`__

.. rubric:: setLayout

.. container:: m-4

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:134 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L134>`__

