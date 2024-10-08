Interface CompiledModel
=======================

.. code-block:: ts

   interface CompiledModel {
       inputs: Output[];
       outputs: Output[];
       getProperty(propertyName): string | number | boolean;
       createInferRequest(): InferRequest;
       exportModelSync(): Buffer;
       input(): Output;
       input(index): Output;
       input(name): Output;
       output(): Output;
       output(index): Output;
       output(name): Output;
       setProperty(properties: {[propertyName: string]: string | number | boolean}): void;
   }

CompiledModel represents a model that is compiled for a specific device by applying
multiple optimization transformations, then mapping to compute kernels.

* **Defined in:**
  `addon.ts:303 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L303>`__


Properties
#####################


.. rubric:: inputs

*

   .. code-block:: ts

      inputs: Output []

   It gets all inputs of a compiled model.

   -  **Defined in:**
      `addon.ts:305 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L305>`__


.. rubric:: outputs

*

   .. code-block:: ts

      outputs: Output []

   It gets all outputs of a compiled model.

   -  **Defined in:**
      `addon.ts:307 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L307>`__


Methods
#####################


.. rubric:: getProperty
   :name: getProperty

*

   .. code-block:: ts

      getProperty(propertyName): string | number | boolean

   It gets the property for the current compiled model.

   * **Parameters:**

     - propertyName: string

       A string to get the property value.

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:313 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L313>`__



.. rubric:: createInferRequest
   :name: createInferRequest

*

   .. code-block:: ts

      createInferRequest(): InferRequest

   It creates an inference request object used to infer the compiled model.

   * **Returns:** :doc:`InferRequest <InferRequest>`

   -  **Defined in:**
      `addon.ts:318 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L318>`__


.. rubric:: exportModelSync
   :name: exportModelSync

*

   .. code-block:: ts

      exportModelSync(): Buffer

   * **Returns:** Buffer

   -  **Defined in:**
      `addon.ts:325 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L325>`__


.. rubric:: input

*

   .. code-block:: ts

      input(): Output

   It gets a single input of a compiled model. If a model has more than one input,
   this method throws an exception.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:349 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L349>`__


   .. code-block:: ts

      input(index): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - index: number

       An input tensor index.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:355 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L355>`__


   .. code-block:: ts

      input(name): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - name: string

       An input tensor name.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:361 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L361>`__


.. rubric:: output

*

   .. code-block:: ts

      output(): Output

   It gets a single output of a compiled model. If a model has more than one output, this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

     A compiled model output.

   * **Defined in:**
     `addon.ts:331 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L331>`__


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
     `addon.ts:337 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L337>`__


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
     `addon.ts:343 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L343>`__


.. rubric:: setProperty
   :name: setProperty

*

   .. code-block:: ts

      setProperty(properties): void

   It sets properties for the current compiled model. Properties can be retrieved via
   :ref:`CompiledModel.getProperty <getProperty>`

   * **Parameters:**

     -

       .. code-block:: ts

          properties: {
                    [propertyName: string]: string | number | boolean;
           }

       An object with the key-value pairs (property name, property value).

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:368 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L368>`__

