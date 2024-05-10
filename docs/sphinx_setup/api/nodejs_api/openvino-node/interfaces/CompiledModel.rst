Interface CompiledModel
=======================

.. code-block:: ts

   interface CompiledModel {
       inputs: Output[];
       outputs: Output[];
       createInferRequest(): InferRequest;
       exportModelSync(): Buffer;
       input(nameOrId?): Output;
       output(nameOrId?): Output;
   }

* **Defined in:**
  `addon.ts:65 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L65>`__


Properties
#####################


.. rubric:: inputs

.. container:: m-4

   .. code-block:: ts

      inputs: Output []

   -  **Defined in:**
      `addon.ts:67 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L67>`__


.. rubric:: outputs

.. container:: m-4

   .. code-block:: ts

      outputs: Output []

   -  **Defined in:**
      `addon.ts:66 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L66>`__


Methods
#####################


.. rubric:: createInferRequest

.. container:: m-4

   .. code-block:: ts

      createInferRequest(): InferRequest

   * **Returns:** :doc:`InferRequest <InferRequest>`

   -  **Defined in:**
      `addon.ts:84 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L70>`__


.. rubric:: exportModelSync

.. container:: m-4

   .. code-block:: ts

      exportModelSync(): Buffer

   * **Returns:** Buffer

   -  **Defined in:**
      `addon.ts:85 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L71>`__


.. rubric:: input

.. container:: m-4

   .. code-block:: ts

      input(nameOrId?): Output


   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          nameOrId: string|number

   * **Returns:** :doc:`InferRequest <Output>`

   * **Defined in:**
     `addon.ts:83 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L69>`__


.. rubric:: output

.. container:: m-4

   .. code-block:: ts

      output(nameOrId?): Output

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          nameOrId: string|number

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
   `addon.ts:82 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L68>`__

