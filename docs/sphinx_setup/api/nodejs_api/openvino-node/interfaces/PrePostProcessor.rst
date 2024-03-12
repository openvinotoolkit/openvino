Interface PrePostProcessor
==========================

.. code-block:: ts

   interface PrePostProcessor {
       build(): PrePostProcessor;
       input(idxOrTensorName?): InputInfo;
       output(idxOrTensorName?): OutputInfo;
   }

- Defined in
  `addon.ts:126 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L126>`__

Methods
#####################

.. rubric:: build


.. code-block:: ts

   build(): PrePostProcessor

**Returns** :doc: `PrePostProcessor <PrePostProcessor>`

- Defined in
  `addon.ts:127 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L127>`__

.. rubric:: input



.. code-block:: ts

   input(idxOrTensorName?): InputInfo

**Parameters**


- ``Optional``

.. code-block:: ts

   idxOrTensorName: string|number


**Returns**  :doc:`InputInfo <InputInfo>`

- Defined in
  `addon.ts:128 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L128>`__

.. rubric:: output


.. code-block:: ts

   output(idxOrTensorName?): OutputInfo


**Parameters**

- ``Optional``

  .. code-block:: ts

     idxOrTensorName: string|number


**Returns**  :doc:`OutputInfo <OutputInfo>`

- Defined in
  `addon.ts:129 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L129>`__
