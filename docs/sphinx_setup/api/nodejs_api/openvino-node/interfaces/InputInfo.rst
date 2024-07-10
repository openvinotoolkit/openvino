Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:542 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L542>`__

Methods
#####################

.. rubric:: model

*

   .. code-block:: ts

      model(): InputModelInfo

   * **Returns:** :doc:`InputModelInfo <InputModelInfo>`

   * **Defined in:**
     `addon.ts:545 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L545>`__


.. rubric:: preprocess

*

   .. code-block:: ts

      preprocess(): PreProcessSteps

   * **Returns:** :doc:`PreProcessSteps <PreProcessSteps>`

   * **Defined in:**
     `addon.ts:544 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L544>`__


.. rubric:: tensor

*

   .. code-block:: ts

      tensor(): InputTensorInfo

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:543 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L543>`__

