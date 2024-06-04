Interface PrePostProcessor
==========================

.. code-block:: ts

   interface PrePostProcessor {
       build(): PrePostProcessor;
       input(idxOrTensorName?): InputInfo;
       output(idxOrTensorName?): OutputInfo;
   }

* **Defined in:**
  `addon.ts:154 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L154>`__


Methods
#####################


.. rubric:: build

.. container:: m-4

   .. code-block:: ts

      build(): PrePostProcessor

   * **Returns:** :doc:`PrePostProcessor <PrePostProcessor>`

   * **Defined in:**
     `addon.ts:155 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L155>`__


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
     `addon.ts:156 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L156>`__

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
     `addon.ts:157 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L157>`__

