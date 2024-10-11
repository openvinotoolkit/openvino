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

* **Defined in:**
  `addon.ts:566 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L566>`__


Properties
#####################


.. rubric:: anyName


*

   .. code-block:: ts

      anyName: string

   -  **Defined in:**
      `addon.ts:567 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L567>`__



.. rubric:: shape

*

   .. code-block:: ts

      shape: number[]

   -  **Defined in:**
      `addon.ts:568 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L568>`__


Methods
#####################


.. rubric:: getAnyName

*

   .. code-block:: ts

     getAnyName(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:570 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L570>`__

.. rubric:: getPartialShape

*

   .. code-block:: ts

       getPartialShape(): PartialShape

   * **Returns:** :doc:`PartialShape <PartialShape>`

   * **Defined in:**
     `addon.ts:572 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L572>`__

.. rubric:: getShape

*

   .. code-block:: ts

      getShape(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:571 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L571>`__

.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:569 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L569>`__

