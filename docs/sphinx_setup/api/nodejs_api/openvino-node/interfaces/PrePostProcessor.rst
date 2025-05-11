Interface PrePostProcessor
==========================

.. code-block:: ts

   interface PrePostProcessor {
       build(): PrePostProcessor;
       input(idxOrTensorName?): InputInfo;
       output(idxOrTensorName?): OutputInfo;
   }

* **Defined in:**
  `addon.ts:621 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L621>`__


Methods
#####################


.. rubric:: build

*

   .. code-block:: ts

      build(): PrePostProcessor

   * **Returns:** :doc:`PrePostProcessor <PrePostProcessor>`

   * **Defined in:**
     `addon.ts:622 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L622>`__


.. rubric:: input

*

   .. code-block:: ts

      input(idxOrTensorName?): InputInfo

   * * **Parameters:**

     - ``Optional``

     .. code-block:: ts

        idxOrTensorName: string|number

   * **Returns:**  :doc:`InputInfo <InputInfo>`

   * **Defined in:**
     `addon.ts:623 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L623>`__

.. rubric:: output

*

   .. code-block:: ts

      output(idxOrTensorName?): OutputInfo

   * **Parameters:**

   - ``Optional``

     .. code-block:: ts

        idxOrTensorName: string|number

   * **Returns:**  :doc:`OutputInfo <OutputInfo>`

   * **Defined in:**
     `addon.ts:624 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L624>`__

