Interface TensorConstructor
===========================

.. code-block:: ts

   interface TensorConstructor {
       new Tensor(type, shape, tensorData?): Tensor;
   }

* **Defined in:**
  `addon.ts:94 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L94>`__


Constructors
#####################


.. rubric:: constructor

.. container:: m-4

   .. code-block:: ts

      new Tensor(type, shape, tensorData?): Tensor

   * **Parameters:**

     - type: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`
     - shape: number[]
     - ``Optional``

       .. code-block:: ts

          tensorData: number[]|SupportedTypedArray

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:95 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L95>`__

