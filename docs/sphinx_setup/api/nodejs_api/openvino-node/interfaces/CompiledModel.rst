Interface CompiledModel
=======================

.. code-block:: ts

   interface CompiledModel {
       inputs: Output[];
       outputs: Output[];
       createInferRequest(): InferRequest;
       input(nameOrId?): Output;
       output(nameOrId?): Output;
   }

- Defined in
  `addon.ts:52 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L52>`__


Properties
#####################

.. rubric:: inputs



.. code-block:: ts

   inputs: Output []

-  Defined in
   `addon.ts:54 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L54>`__



.. rubric:: outputs



.. code-block:: ts

   outputs: Output []

-  Defined in
   `addon.ts:53 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L53>`__



Methods
#####################

.. rubric:: createInferRequest


.. code-block:: ts

   createInferRequest(): InferRequest

**Returns** :doc:`InferRequest <InferRequest>`

-  Defined in
   `addon.ts:57 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L57>`__



.. rubric:: input



.. code-block:: ts

   input(nameOrId?): Output


**Parameters**

- ``Optional``

  .. code-block:: ts

     nameOrId: string|number

**Returns** :doc:`InferRequest <Output>`

- Defined in
  `addon.ts:56 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L56>`__



.. rubric:: output


.. code-block:: ts

   output(nameOrId?): Output

- ``Optional``

  .. code-block:: ts

     nameOrId: string|number

**Returns**  :doc:`Output <Output>`



- Defined in
  `addon.ts:55 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L55>`__

