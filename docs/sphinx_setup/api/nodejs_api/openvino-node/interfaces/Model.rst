Interface Model
===============

.. code-block:: ts

   interface Model {
       clone(): Model;
       inputs: Output[];
       outputs: Output[];
       getFriendlyName(): string;
       getName(): string;
       getOutputShape(index): number[];
       getOutputSize(): number;
       getOutputElementType(index): string;
       input(): Output;
       input(name): Output;
       input(index): Output;
       isDynamic(): boolean;
       output(): Output;
       output(name): Output;
       output(index): Output;
       setFriendlyName(name): void;
   }

A user-defined model read by :ref:`Core.readModel <readModel>`.

* **Defined in:**
  `addon.ts:230 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L230>`__


Properties
#####################


.. rubric:: inputs

*

   .. code-block:: ts

      inputs: Output[]

   -  **Defined in:**
      `addon.ts:305 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L305>`__

.. rubric:: outputs


*

   .. code-block:: ts

      outputs: Output[]

   -  **Defined in:**
      `addon.ts:309 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L309>`__


Methods
#####################

.. rubric:: clone
   :name: clone

*

   .. code-block:: ts

      clone(): Model;

   It returns a cloned model.

   * **Returns:** :doc:`Model <Model>`

   * **Defined in:**
     `addon.ts:234 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L234>`__


.. rubric:: getFriendlyName
   name: getFriendlyName

*

   .. code-block:: ts

      getFriendlyName(): string

   It gets the friendly name for a model. If a friendly name is not set
   via :ref:`Model.setFriendlyName <setFriendlyName>`, a unique model name is returned.

   * **Returns:** string

     A string with a friendly name of the model.

   * **Defined in:**
     `addon.ts:240 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L240>`__


.. rubric:: getName

*

   .. code-block:: ts

      getName(): string

   It gets the unique name of the model.

   * **Returns:** string

     A string with the name of the model.

   * **Defined in:**
     `addon.ts:245 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L245>`__


.. rubric:: getOutputShape

*

   .. code-block:: ts

      getOutputShape(): number[]

   It returns the shape of the element at the specified index.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:250 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L250>`__


.. rubric:: getOutputSize

*

   .. code-block:: ts

      getOutputSize(): number[]

   It returns the number of the model outputs.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:254 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L254>`__

.. rubric:: getOutputElementType
   :name: getOutputElementType

*

   .. code-block:: ts

      getOutputElementType(index): string;

   It gets the element type of a specific output of the model.

   * **Parameters:**

     -

       .. code-block:: ts

          index: number

       The index of the output.

   * **Returns:** string

   * **Defined in:**
     `addon.ts:259 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L259>`__


.. rubric:: input

*

   .. code-block:: ts

      input(): Output

   It gets the input of the model. If a model has more than one input,
   this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:264 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L264>`__


   .. code-block:: ts

      input(name: string): Output

   It gets the input of the model identified by the tensor name.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          name: string

       The tensor name.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:269 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L269>`__


   .. code-block:: ts

      input(index: number): Output

   It gets the input of the model identified by the index.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          index: number

       The index of the input.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:274 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L274>`__


.. rubric:: isDynamic

*

   .. code-block:: ts

      isDynamic(): boolean

   It returns true if any of the ops defined in the model contains a partial shape.

   * **Returns:**  boolean

   * **Defined in:**
     `addon.ts:279 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L279>`__


.. rubric:: output

*

   .. code-block:: ts

      output(): Output

   It gets the output of the model.
   If a model has more than one output, this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:284 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L284>`__


.. rubric:: output

*

   .. code-block:: ts

      output(name): Output

   It gets the output of the model identified by the tensor name.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          name: string

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:289 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L289>`__


   .. code-block:: ts

      output(index): Output

   It gets the output of the model identified by the index.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          index: number

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:294 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L294>`__


.. rubric:: setFriendlyName
   :name: setFriendlyName

*

   .. code-block:: ts

      setFriendlyName(name): void

   Sets a friendly name for the model. This does not overwrite the unique
   model name and is retrieved via :ref:`Model.getFriendlyName <getFriendlyName>`.
   Mainly used for debugging.

   * **Parameters:**

     - name: string

   * **Returns:** void

   * **Defined in:**
     `addon.ts:301 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L301>`__