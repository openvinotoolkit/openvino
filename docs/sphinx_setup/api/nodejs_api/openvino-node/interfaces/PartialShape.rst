Interface PartialShape
======================

.. code-block:: ts

   interface PartialShape {
       getDimensions(): Dimension[];
       isDynamic(): boolean;
       isStatic(): boolean;
       toString(): string;
   }

- Defined in
  `addon.ts:135 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L135>`__

Methods
#####################

.. rubric:: getDimensions


.. code-block:: ts

   getDimensions(): Dimension


**Returns** :doc:`Dimension <../types/Dimension>` []

- Defined in
  `addon.ts:139 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L139>`__

.. rubric:: isDynamic


.. code-block:: ts

   isDynamic(): boolean


**Returns** boolean

- Defined in
  `addon.ts:137 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L137>`__

.. rubric:: isStatic



.. code-block:: ts

   isStatic(): boolean


**Returns** boolean


- Defined in
  `addon.ts:136 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L136>`__

.. rubric:: toString


.. code-block:: ts

   toString(): string


**Returns** string

- Defined in
  `addon.ts:138 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L138>`__

