Interface InputInfo
===================

.. code-block:: ts

   interface InputInfo {
       model(): InputModelInfo;
       preprocess(): PreProcessSteps;
       tensor(): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:144 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L130>`__

Methods
#####################

.. rubric:: model

.. container:: m-4

   .. code-block:: ts

      model(): InputModelInfo

   * **Returns:** :doc:`InputModelInfo <InputModelInfo>`

   * **Defined in:**
     `addon.ts:147 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L133>`__


.. rubric:: preprocess

.. container:: m-4

   .. code-block:: ts

      preprocess(): PreProcessSteps

   * **Returns:** :doc:`PreProcessSteps <PreProcessSteps>`

   * **Defined in:**
     `addon.ts:146 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L132>`__


.. rubric:: tensor

.. container:: m-4

   .. code-block:: ts

      tensor(): InputTensorInfo

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:145 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L131>`__

