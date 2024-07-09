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
  `addon.ts:515 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L515>`__


Properties
#####################


.. rubric:: anyName


*

   .. code-block:: ts

      anyName: string

   -  **Defined in:**
      `addon.ts:516 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L516>`__



.. rubric:: shape

*

   .. code-block:: ts

      shape: number[]

   -  **Defined in:**
      `addon.ts:517 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L517>`__


Methods
#####################


.. rubric:: getAnyName

*

   .. code-block:: ts

     getAnyName(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:519 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L519>`__

.. rubric:: getPartialShape

*

   .. code-block:: ts

       getPartialShape(): PartialShape

   * **Returns:** :doc:`PartialShape <PartialShape>`

   * **Defined in:**
     `addon.ts:521 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L521>`__

.. rubric:: getShape

*

   .. code-block:: ts

      getShape(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:520 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L520>`__

.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:518 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L518>`__

