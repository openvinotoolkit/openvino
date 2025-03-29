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
       setProperty(properties: Record<string, OVAny>): void;
   }

CompiledModel represents a model that is compiled for a specific device by applying
multiple optimization transformations, then mapping to compute kernels.

* **Defined in:**
  `addon.ts:317 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L317>`__


Properties
#####################


.. rubric:: inputs

*

   .. code-block:: ts

      inputs: Output []

   It gets all inputs of a compiled model.

   -  **Defined in:**
      `addon.ts:319 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L319>`__


.. rubric:: outputs

*

   .. code-block:: ts

      outputs: Output []

   It gets all outputs of a compiled model.

   -  **Defined in:**
      `addon.ts:321 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L321>`__


Methods
#####################


.. rubric:: getProperty
   :name: getProperty

*

   .. code-block:: ts

      getProperty(propertyName): OVAny

   It gets the property for the current compiled model.

   * **Parameters:**

     - propertyName: string

       A string to get the property value.

   * **Returns:**  :doc:`OVAny <../types/OVAny>`

   * **Defined in:**
     `addon.ts:327 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L327>`__



.. rubric:: createInferRequest
   :name: createInferRequest

*

   .. code-block:: ts

      createInferRequest(): InferRequest

   It creates an inference request object used to infer the compiled model.

   * **Returns:** :doc:`InferRequest <InferRequest>`

   -  **Defined in:**
      `addon.ts:332 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L332>`__


.. rubric:: exportModelSync
   :name: exportModelSync

*

   .. code-block:: ts

      exportModelSync(): Buffer

   * **Returns:** Buffer

   -  **Defined in:**
      `addon.ts:339 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L339>`__


.. rubric:: input

*

   .. code-block:: ts

      input(): Output

   It gets a single input of a compiled model. If a model has more than one input,
   this method throws an exception.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:363 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L363>`__


   .. code-block:: ts

      input(index): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - index: number

       An input tensor index.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:369 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L369>`__


   .. code-block:: ts

      input(name): Output

   It gets input of a compiled model identified by an index.

   * **Parameters:**

     - name: string

       An input tensor name.

   * **Returns:** :doc:`Output <Output>`

     A compiled model input.

   * **Defined in:**
     `addon.ts:375 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L375>`__


.. rubric:: output

*

   .. code-block:: ts

      output(): Output

   It gets a single output of a compiled model. If a model has more than one output, this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

     A compiled model output.

   * **Defined in:**
     `addon.ts:345 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L345>`__


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
     `addon.ts:351 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L351>`__


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
     `addon.ts:357 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L357>`__


.. rubric:: setProperty
   :name: setProperty

*

   .. code-block:: ts

      setProperty(properties: Record<string, OVAny>): void

   It sets properties for the current compiled model. Properties can be retrieved via
   :ref:`CompiledModel.getProperty <getProperty>`

   * **Parameters:**

     -

       .. code-block:: ts

          properties: Record<string, OVAny>,

       An object with the key-value pairs (property name, property value).

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:382 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L382>`__

