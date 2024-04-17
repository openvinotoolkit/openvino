InferRequest
============

.. rubric:: Interface InferRequest


.. code-block:: ts

   interface InferRequest {
       getCompiledModel(): CompiledModel;
       getInputTensor(idx?): Tensor;
       getOutputTensor(idx?): Tensor;
       getTensor(nameOrOutput): Tensor;
       infer(inputData?): {
           [outputName: string]: Tensor;
       };
       inferAsync(inputData): Promise<{
           [outputName: string]: Tensor;
       }>;
       setInputTensor(idxOrTensor, tensor?): void;
       setOutputTensor(idxOrTensor, tensor?): void;
       setTensor(name, tensor): void;
   }

- Defined in
  `addon.ts:72 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L72>`__

Methods
#####################


.. rubric:: getCompiledModel

.. code-block:: ts

   getCompiledModel(): CompiledModel

**Returns** :doc:`CompiledModel <CompiledModel>`

- Defined in
  `addon.ts:83 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L83>`__


.. rubric:: getInputTensor

.. code-block:: ts

   getInputTensor(idx?): Tensor


**Parameters**

- ``Optional``

  .. code-block:: ts

     idx: number


**Returns**  :doc:`Tensor <Tensor>`

- Defined in
  `addon.ts:77 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L77>`__

.. rubric:: getOutputTensor

.. code-block:: ts

   getOutputTensor(idx?): Tensor


**Parameters**

- ``Optional``

  .. code-block:: ts

     idx: number

**Returns**  :doc:`Tensor <Tensor>`


- Defined in
  `addon.ts:78 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L78>`__

.. rubric:: getTensor

.. code-block:: ts

   getTensor(nameOrOutput): Tensor

**Parameters**

- nameOrOutput: string| :doc:`Output <Output>`

**Returns**  :doc:`Tensor <Tensor>`

- Defined in
  `addon.ts:76 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L76>`__

.. rubric:: infer


.. code-block:: ts

   infer(inputData?): {
       [outputName: string]: Tensor;
   }


**Parameters**

- ``Optional``

  .. code-block:: ts

     inputData: {
                [inputName: string]: Tensor | SupportedTypedArray;
                } | Tensor[] | SupportedTypedArray[]

**Returns**

.. code-block:: ts

   {
   [outputName: string]: Tensor;
   }

- [outputName: string]: Tensor


- Defined in
  `addon.ts:79 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L79>`__

.. rubric:: inferAsync


.. code-block:: ts

   inferAsync(inputData): Promise<{
       [outputName: string]: Tensor;
   }>

**Parameters**

-

  .. code-block:: ts

     inputData: Tensor[] | {
         [inputName: string]: Tensor;
     }

**Returns**

.. code-block:: ts

   Promise<{
    [outputName: string]: Tensor;
   }>


- Defined in
  `addon.ts:81 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L81>`__

.. rubric:: setInputTensor

.. code-block:: ts

   setInputTensor(idxOrTensor, tensor?): void


**Parameters**

- idxOrTensor: number| :doc:`Tensor <Tensor>`

- ``Optional``

  .. code-block:: ts

     tensor: Tensor


**Returns**  void

- Defined in
  `addon.ts:74 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L74>`__

.. rubric:: setOutputTensor


.. code-block:: ts

   setOutputTensor(idxOrTensor, tensor?): void


**Parameters**

- idxOrTensor: number| :doc:`Tensor <Tensor>`
- ``Optional``

  .. code-block:: ts

     tensor: Tensor


**Returns**  void

- Defined in
  `addon.ts:75 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L75>`__

.. rubric:: setTensor


.. code-block:: ts

   setTensor(name, tensor): void

**Parameters**

- name: string
- tensor: :doc:`Tensor <Tensor>`

**Returns**  void

- Defined in
  `addon.ts:73 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L73>`__
