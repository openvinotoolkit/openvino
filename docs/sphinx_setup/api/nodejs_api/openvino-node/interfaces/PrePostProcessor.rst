Interface PrePostProcessor
==========================

.. code-block:: ts

   interface PrePostProcessor {
       build(): PrePostProcessor;
       input(idxOrTensorName?): InputInfo;
       output(idxOrTensorName?): OutputInfo;
   }

* **Defined in:**
  `addon.ts:603 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L603>`__


Methods
#####################


.. rubric:: build

*

   .. code-block:: ts

      build(): PrePostProcessor

   * **Returns:** :doc:`PrePostProcessor <PrePostProcessor>`

   * **Defined in:**
     `addon.ts:604 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L604>`__


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
     `addon.ts:605 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L605>`__

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
     `addon.ts:606 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L606>`__

