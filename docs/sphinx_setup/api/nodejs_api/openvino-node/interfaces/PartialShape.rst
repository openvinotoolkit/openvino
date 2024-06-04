Interface PartialShape
======================

.. code-block:: ts

   interface PartialShape {
       getDimensions(): Dimension[];
       isDynamic(): boolean;
       isStatic(): boolean;
       toString(): string;
   }

* **Defined in:**
  `addon.ts:163 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L163>`__


Methods
#####################


.. rubric:: getDimensions

.. container:: m-4

  .. code-block:: ts

     getDimensions(): Dimension

  * **Returns:** :doc:`Dimension <../types/Dimension>`\[]

  * **Defined in:**
    `addon.ts:167 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L167>`__


.. rubric:: isDynamic

.. container:: m-4

   .. code-block:: ts

      isDynamic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:165 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L165>`__


.. rubric:: isStatic

.. container:: m-4

   .. code-block:: ts

      isStatic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:164 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L164>`__


.. rubric:: toString

.. container:: m-4

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:166 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L166>`__

