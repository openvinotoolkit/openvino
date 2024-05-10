Interface Model
===============

.. code-block:: ts

   interface Model {
       inputs: Output[];
       outputs: Output[];
       getName(): string;
       input(nameOrId?): Output;
       isDynamic(): boolean;
       output(nameOrId?): Output;
   }

* **Defined in:**
  `addon.ts:68 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L56>`__


Properties
#####################


.. rubric:: inputs

.. container:: m-4

   .. code-block:: ts

      inputs: Output[]

   -  **Defined in:**
      `addon.ts:70 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L58>`__

.. rubric:: outputs


.. container:: m-4

   .. code-block:: ts

      outputs: Output[]

   -  **Defined in:**
      `addon.ts:69 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L57>`__


Methods
#####################


.. rubric:: getName

.. container:: m-4

   .. code-block:: ts

      getName(): string


   * **Returns:** string

   * **Defined in:**
     `addon.ts:73 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L61>`__


.. rubric:: input

.. container:: m-4

   .. code-block:: ts

      input(nameOrId?): Output

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          nameOrId: string|number

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:72 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L60>`__


.. rubric:: isDynamic

.. container:: m-4

   .. code-block:: ts

      isDynamic(): boolean

   * **Returns:**  boolean

   * **Defined in:**
     `addon.ts:74 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L62>`__


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
     `addon.ts:71 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/bindings/js/node/lib/addon.ts#L59>`__

