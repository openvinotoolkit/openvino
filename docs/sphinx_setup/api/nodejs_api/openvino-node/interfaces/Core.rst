Interface Core
==============

.. code-block:: json

   interface Core {
       compileModel(model, device, config?): Promise<CompiledModel>;
       compileModelSync(model, device, config?): CompiledModel;
       readModel(modelPath, weightsPath?): Promise<Model>;
       readModel(modelBuffer, weightsBuffer?): Promise<Model>;
       readModelSync(modelPath, weightsPath?): Model;
       readModelSync(modelBuffer, weightsBuffer?): Model;
   }}

- Defined in
  `addon.ts:23 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L23>`__


Methods
#####################


.. rubric:: compileModel


.. code-block:: json

   compileModel(model, device, config?): Promise<CompiledModel>


**Parameters**


-  model: :doc:`Model <Model>`
-  device: string
- ``Optional``

  .. code-block:: json

     config: {
         [option: string]:string;
     }


- [option: string]:string


**Returns** Promise<\ :doc:`CompiledModel <CompiledModel>` \>

- Defined in
  `addon.ts:24 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L24>`__


.. rubric:: compileModelSync


.. code-block:: json

   compileModelSync(model, device, config?):CompiledModel


**Parameters**

- model: :doc:`Model <Model>`
- device: string
- ``Optional``

  .. code-block:: json

     config: {
               [option: string]:string;
      }

- [option: string]:string


**Returns** :doc:`CompiledModel <CompiledModel>`


- Defined in
  `addon.ts:29 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L29>`__


.. rubric:: readModel


.. code-block:: json

   readModel(modelPath, weightsPath?): Promise<Model>


**Parameters**

 - modelPath: string
 - ``Optional``

  .. code-block:: json

     weightsPath: string


**Returns**  Promise<\ :doc:`Model <Model>`\ >

- Defined in
  `addon.ts:34 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L34>`__

.. code-block:: json

   readModel(modelBuffer, weightsBuffer?): Promise<Model>

**Parameters**

- modelBuffer: Uint8Array
- ``Optional``

  .. code-block:: json

     weightsBuffer: Uint8Array


**Returns**  Promise<\ :doc:`Model <Model>`\ >


- Defined in
  `addon.ts:35 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L35>`__

.. rubric:: readModelSync


.. code-block:: json

   readModelSync(modelPath, weightsPath?): Model


**Parameters**

- modelPath: string
- ``Optional``

  .. code-block:: json

     weightsPath: string

**Returns**  :doc:`Model <Model>`

- Defined in
  `addon.ts:37 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L37>`__

.. code-block:: json

   readModelSync(modelBuffer, weightsBuffer?): Model


**Parameters**

- modelBuffer: Uint8Array
- ``Optional``

  .. code-block:: json

     weightsBuffer: Uint8Array

**Returns**  :doc:`Model <Model>`

- Defined in
  `addon.ts:38 <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/bindings/js/node/lib/addon.ts#L38>`__
