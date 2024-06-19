Interface PrePostProcessor
==========================

.. code-block:: ts

   interface PrePostProcessor {
       build(): PrePostProcessor;
       input(idxOrTensorName?): InputInfo;
       output(idxOrTensorName?): OutputInfo;
   }

* **Defined in:**
  `addon.ts:552 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L552>`__


Methods
#####################


.. rubric:: build

.. container:: m-4

   .. code-block:: ts

      build(): PrePostProcessor

   * **Returns:** :doc:`PrePostProcessor <PrePostProcessor>`

   * **Defined in:**
     `addon.ts:553 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L553>`__


.. rubric:: input

.. container:: m-4

   .. code-block:: ts

      input(idxOrTensorName?): InputInfo

   * * **Parameters:**

     - ``Optional``

     .. code-block:: ts

        idxOrTensorName: string|number

   * **Returns:**  :doc:`InputInfo <InputInfo>`

   * **Defined in:**
     `addon.ts:554 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L554>`__

.. rubric:: output

.. container:: m-4

   .. code-block:: ts

      output(idxOrTensorName?): OutputInfo

   * **Parameters:**

   - ``Optional``

     .. code-block:: ts

        idxOrTensorName: string|number

   * **Returns:**  :doc:`OutputInfo <OutputInfo>`

   * **Defined in:**
     `addon.ts:555 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L555>`__

