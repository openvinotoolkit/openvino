Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:611 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L611>`__

Methods
#####################

.. rubric:: model

*

   .. code-block:: ts

      model(): InputModelInfo

   * **Returns:** :doc:`InputModelInfo <InputModelInfo>`

   * **Defined in:**
     `addon.ts:614 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L614>`__


.. rubric:: preprocess

*

   .. code-block:: ts

      preprocess(): PreProcessSteps

   * **Returns:** :doc:`PreProcessSteps <PreProcessSteps>`

   * **Defined in:**
     `addon.ts:613 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L613>`__


.. rubric:: tensor

*

   .. code-block:: ts

      tensor(): InputTensorInfo

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:612 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L612>`__

