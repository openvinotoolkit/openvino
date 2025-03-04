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
  `addon.ts:584 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L584>`__


Properties
#####################


.. rubric:: anyName


*

   .. code-block:: ts

      anyName: string

   -  **Defined in:**
      `addon.ts:585 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L585>`__



.. rubric:: shape

*

   .. code-block:: ts

      shape: number[]

   -  **Defined in:**
      `addon.ts:586 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L586>`__


Methods
#####################


.. rubric:: getAnyName

*

   .. code-block:: ts

     getAnyName(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:588 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L588>`__

.. rubric:: getPartialShape

*

   .. code-block:: ts

       getPartialShape(): PartialShape

   * **Returns:** :doc:`PartialShape <PartialShape>`

   * **Defined in:**
     `addon.ts:590 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L590>`__

.. rubric:: getShape

*

   .. code-block:: ts

      getShape(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:589 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L589>`__

.. rubric:: toString

*

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:587 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L587>`__

