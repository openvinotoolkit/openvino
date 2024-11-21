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
  `addon.ts:630 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L630>`__


Methods
#####################


.. rubric:: getDimensions

*

  .. code-block:: ts

     getDimensions(): Dimension

  * **Returns:** :doc:`Dimension <../types/Dimension>`\[]

  * **Defined in:**
    `addon.ts:634 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L634>`__


.. rubric:: isDynamic

*

   .. code-block:: ts

      isDynamic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:632 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L632>`__


.. rubric:: isStatic

*

   .. code-block:: ts

      isStatic(): boolean

   * **Returns:** boolean

   * **Defined in:**
     `addon.ts:631 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L631>`__


.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:633 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L633>`__

