Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:593 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L593>`__

Methods
#####################

.. rubric:: model

*

   .. code-block:: ts

      model(): InputModelInfo

   * **Returns:** :doc:`InputModelInfo <InputModelInfo>`

   * **Defined in:**
     `addon.ts:596 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L596>`__


.. rubric:: preprocess

*

   .. code-block:: ts

      preprocess(): PreProcessSteps

   * **Returns:** :doc:`PreProcessSteps <PreProcessSteps>`

   * **Defined in:**
     `addon.ts:595 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L595>`__


.. rubric:: tensor

*

   .. code-block:: ts

      tensor(): InputTensorInfo

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:594 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L594>`__

