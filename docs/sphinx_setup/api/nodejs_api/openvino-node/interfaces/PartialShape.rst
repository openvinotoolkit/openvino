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
  `addon.ts:612 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L612>`__


Methods
#####################


.. rubric:: getDimensions

*

  .. code-block:: ts

     getDimensions(): Dimension

  * **Returns:** :doc:`Dimension <../types/Dimension>`\[]

  * **Defined in:**
    `addon.ts:616 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L616>`__


.. rubric:: isDynamic

*

   .. code-block:: ts

      isDynamic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:614 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L614>`__


.. rubric:: isStatic

*

   .. code-block:: ts

      isStatic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:613 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L613>`__


.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:615 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L615>`__

