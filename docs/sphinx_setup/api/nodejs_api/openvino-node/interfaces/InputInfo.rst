Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:144 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L144>`__

Methods
#####################

.. rubric:: model

.. container:: m-4

   .. code-block:: ts

      model(): InputModelInfo

   * **Returns:** :doc:`InputModelInfo <InputModelInfo>`

   * **Defined in:**
     `addon.ts:147 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L147>`__


.. rubric:: preprocess

.. container:: m-4

   .. code-block:: ts

      preprocess(): PreProcessSteps

   * **Returns:** :doc:`PreProcessSteps <PreProcessSteps>`

   * **Defined in:**
     `addon.ts:146 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L146>`__


.. rubric:: tensor

.. container:: m-4

   .. code-block:: ts

      tensor(): InputTensorInfo

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:145 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L145>`__

