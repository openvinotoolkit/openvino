Interface CompiledModel
=======================

.. code-block:: ts

   interface CompiledModel {
       inputs: Output[];
       outputs: Output[];
       createInferRequest(): InferRequest;
       exportModelSync(): Buffer;
       input(): Output;
       input(index): Output;
       input(name): Output;
       output(): Output;
       output(index): Output;
       output(name): Output;
   }

CompiledModel represents a model that is compiled for a specific device by applying
multiple optimization transformations, then mapping to compute kernels.

* **Defined in:**
  `addon.ts:272 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L272>`__


Properties
#####################


.. rubric:: inputs

*

   .. code-block:: ts

      inputs: Output []

   It gets all inputs of a compiled model.

   -  **Defined in:**
      `addon.ts:274 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L274>`__


.. rubric:: outputs

*

   .. code-block:: ts

      outputs: Output []

   It gets all outputs of a compiled model.

   -  **Defined in:**
      `addon.ts:276 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L276>`__


Methods
#####################


.. rubric:: createInferRequest
   :name: createInferRequest

*

   .. code-block:: ts

      createInferRequest(): InferRequest

   It creates an inference request object used to infer the compiled model.

   * **Returns:** :doc:`InferRequest <InferRequest>`

   -  **Defined in:**
      `addon.ts:281 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L281>`__


.. rubric:: exportModelSync
   :name: exportModelSync

*

   .. code-block:: ts

      exportModelSync(): Buffer

   * **Returns:** Buffer

   -  **Defined in:**
      `addon.ts:288 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L288>`__


.. rubric:: input

*

   .. code-block:: ts

      input(): Output

   It gets a single input of a compiled model. If a model has more than one input,
   this method throws an exception.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:312 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L312>`__


   .. code-block:: ts

      input(index): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - index: number

       An input tensor index.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:318 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L318>`__


   .. code-block:: ts

      input(name): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - name: string

       An input tensor name.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:324 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L324>`__


.. rubric:: output

*

   .. code-block:: ts

      output(): Output

   It gets a single output of a compiled model. If a model has more than one output, this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

     A compiled model output.

   * **Defined in:**
     `addon.ts:294 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L294>`__


   .. code-block:: ts

      output(index): Output

   It gets output of a compiled model identified by an index.

   * **Parameters:**

     -

       .. code-block:: ts

          index: number

       An output tensor index.

   * **Returns:**  :doc:`Output <Output>`

     A compiled model output.

   * **Defined in:**
     `addon.ts:300 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L300>`__


   .. code-block:: ts

      output(name): Output

   It gets output of a compiled model identified by a tensorName.

   * **Parameters:**

     -

       .. code-block:: ts

          name: string

       An output tensor name.

   * **Returns:**  :doc:`Output <Output>`

     A compiled model output.

   * **Defined in:**
     `addon.ts:306 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L306>`__

