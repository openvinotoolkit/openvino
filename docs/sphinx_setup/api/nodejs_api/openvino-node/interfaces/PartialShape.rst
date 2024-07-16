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
  `addon.ts:561 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L561>`__


Methods
#####################


.. rubric:: getDimensions

*

  .. code-block:: ts

     getDimensions(): Dimension

  * **Returns:** :doc:`Dimension <../types/Dimension>`\[]

  * **Defined in:**
    `addon.ts:565 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L565>`__


.. rubric:: isDynamic

*

   .. code-block:: ts

      isDynamic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:563 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L563>`__


.. rubric:: isStatic

*

   .. code-block:: ts

      isStatic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:562 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L562>`__


.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:564 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L564>`__

