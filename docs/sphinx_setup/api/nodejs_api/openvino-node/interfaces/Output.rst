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
  `addon.ts:117 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L117>`__


Properties
#####################


.. rubric:: anyName


.. container:: m-4

   .. code-block:: ts

      anyName: string

   -  **Defined in:**
      `addon.ts:118 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L118>`__



.. rubric:: shape

.. container:: m-4

   .. code-block:: ts

      shape: number[]

   -  **Defined in:**
      `addon.ts:119 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L119>`__


Methods
#####################


.. rubric:: getAnyName

.. container:: m-4

   .. code-block:: ts

     getAnyName(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:121 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L121>`__

.. rubric:: getPartialShape

.. container:: m-4

   .. code-block:: ts

       getPartialShape(): PartialShape

   * **Returns:** :doc:`PartialShape <PartialShape>`

   * **Defined in:**
     `addon.ts:123 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L123>`__

.. rubric:: getShape

.. container:: m-4

   .. code-block:: ts

      getShape(): number[]

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:122 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L122>`__

.. rubric:: toString

.. container:: m-4

   .. code-block:: ts

      toString(): string

   * **Returns:** string

   * **Defined in:**
     `addon.ts:120 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L120>`__

