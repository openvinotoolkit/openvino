Interface Output
================


.. code-block:: ts

   interface Output {
       anyName: string;
       shape: number[];
       getAnyName(): string;
       getPartialShape(): PartialShape;
       getShape(): number[];
       toString(): string;
   }

- Defined in
  `addon.ts:89 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L89>`__

Properties
#####################

.. rubric:: anyName



.. code-block:: ts

   anyName: string

-  Defined in
   `addon.ts:90 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L90>`__



.. rubric:: shape



.. code-block:: ts

   shape: number[]

-  Defined in
   `addon.ts:91 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L91>`__



Methods
#####################

.. rubric:: getAnyName


.. code-block:: ts

  getAnyName(): string


**Returns** string

- Defined in
  `addon.ts:93 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L93>`__

.. rubric:: getPartialShape


.. code-block:: ts

    getPartialShape(): PartialShape


**Returns** :doc:`PartialShape <PartialShape>`

- Defined in
  `addon.ts:95 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L95>`__

.. rubric:: getShape


.. code-block:: ts

   getShape(): number[]

**Returns**

.. code-block:: ts

   number[]

- Defined in
  `addon.ts:94 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L94>`__

.. rubric:: toString


.. code-block:: ts

   toString(): string

**Returns**

.. code-block:: ts

   string

- Defined in
  `addon.ts:92 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L92>`__
