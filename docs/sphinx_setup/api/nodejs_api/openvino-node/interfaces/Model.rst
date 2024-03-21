Interface Model
===============

.. rubric:: Interface Model


.. code-block:: ts

   interface Model {
       inputs: Output[];
       outputs: Output[];
       getName(): string;
       input(nameOrId?): Output;
       output(nameOrId?): Output;
   }

- Defined in
  `addon.ts:44 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L44>`__


Properties
#####################


.. rubric:: inputs



.. code-block:: ts

   inputs: Output[]

-  Defined in
   `addon.ts:46 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L46>`__

.. rubric:: outputs



.. code-block:: ts

   outputs: Output[]

-  Defined in
   `addon.ts:45 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L45>`__


Methods
#####################


.. rubric:: getName


.. code-block:: ts

   getName(): string


**Returns** string

- Defined in
  `addon.ts:49 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L49>`__


.. rubric:: input


.. code-block:: ts

   input(nameOrId?): Output


**Parameters**


- ``Optional``

  .. code-block:: ts

     nameOrId: string|number


**Returns**  :doc:`Output <Output>`


- Defined in
  `addon.ts:48 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L48>`__


.. rubric:: output


.. code-block:: ts

   output(nameOrId?): Output


**Parameters**

- ``Optional``

  .. code-block:: ts

     nameOrId: string|number

**Returns**  :doc:`Output <Output>`

- Defined in
  `addon.ts:47 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L47>`__
