Interface Tensor
=====================

.. code-block:: ts

   interface Tensor {
       data: number[];
       getData(): number[];
       getElementType(): element;
       getShape(): number[];
       getSize(): number;
   }

* **Defined in:**
  `addon.ts:88 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L88>`__


Properties
#####################


.. rubric:: data

.. container:: m-4

   .. code-block:: ts

      data: SupportedTypedArray

   -  **Defined in:**
      `addon.ts:89 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L89>`__


Methods
#####################


.. rubric:: getData

.. container:: m-4

   .. code-block:: ts

      getData(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:92 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L92>`__

.. rubric:: getElementType

.. container:: m-4

   .. code-block:: ts

      getElementType(): element

   * **Returns:** :doc:`element <../enums/element>`

   * **Defined in:**
     `addon.ts:90 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L90>`__


.. rubric:: getShape

.. container:: m-4

   .. code-block:: ts

      getShape(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:91 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L91>`__


.. rubric:: getSize

.. container:: m-4

   .. code-block:: ts

      getSize(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:93 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L93>`__

