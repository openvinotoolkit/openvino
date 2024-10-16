InferRequest
============

.. code-block:: ts

   interface InferRequest {
       getCompiledModel(): CompiledModel;
       getInputTensor(): Tensor;
       getInputTensor(idx): Tensor;
       getOutputTensor(): Tensor;
       getOutputTensor(idx?): Tensor;
       getTensor(nameOrOutput): Tensor;
       infer(): {
           [outputName: string]: Tensor;
       };
       infer(inputData): {
           [outputName: string]: Tensor;
       };
       infer(inputData): {
           [outputName: string]: Tensor;
       };
       inferAsync(inputData): Promise<{
           [outputName: string]: Tensor;
       }>;
       setInputTensor(tensor): void;
       setInputTensor(idx, tensor): void;
       setOutputTensor(tensor): void;
       setOutputTensor(idx, tensor): void;
       setTensor(name, tensor): void;
   }


The ``InferRequest`` object is created using
:ref:`CompiledModel.createInferRequest <createInferRequest>` method and is
specific for a given deployed model. It is used to make predictions and
can be run in asynchronous or synchronous manners.


* **Defined in:**
  `addon.ts:453 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L453>`__


Methods
#####################


.. rubric:: getCompiledModel

*

   .. code-block:: ts

      getCompiledModel(): CompiledModel

   It gets the compiled model used by the InferRequest object.

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:490 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L490>`__


.. rubric:: getInputTensor

*

   .. code-block:: ts

      getInputTensor(): Tensor

   It gets the input tensor for inference.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          idx: number


   * **Returns:**  :doc:`Tensor <Tensor>`

     The input tensor for the model. If the model has several inputs,
     an exception is thrown.

   * **Defined in:**
     `addon.ts:496 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L496>`__


   .. code-block:: ts

      getInputTensor(idx: number): Tensor

   It gets the input tensor for inference.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          idx: number

       An index of the tensor to get.

   * **Returns:**  :doc:`Tensor <Tensor>`

     A tensor at the specified index. If the tensor with the specified
     idx is not found, an exception is thrown.

   * **Defined in:**
     `addon.ts:503 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L503>`__


.. rubric:: getOutputTensor

*

   .. code-block:: ts

      getOutputTensor(): Tensor

   It gets the output tensor for inference.

   * **Returns:**  :doc:`Tensor <Tensor>`

     The output tensor for the model. If the tensor with the specified
     idx is not found, an exception is thrown.

   * **Defined in:**
     `addon.ts:509 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L509>`__


   .. code-block:: ts

      getOutputTensor(idx?: number): Tensor

   It gets the output tensor for inference.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          idx: number

       An index of the tensor to get.

   * **Returns:**  :doc:`Tensor <Tensor>`

     A tensor at the specified index. If the tensor with the specified
     idx is not found, an exception is thrown.

   * **Defined in:**
     `addon.ts:516 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L516>`__

.. rubric:: getTensor

*

   .. code-block:: ts

      getTensor(nameOrOutput: string | Output): Tensor;

   It gets an input/output tensor for inference.
   If a tensor with the specified name or port is not found, an exception
   is thrown.

   * **Parameters:**

     - nameOrOutput: string | :doc:`Output <Output>`

       The name of the tensor or output object.

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:525 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L525>`__


.. rubric:: infer

*

   .. code-block:: ts

      infer(): { [outputName: string] : Tensor};

   It infers specified input(s) in the synchronous mode. Inputs have to be
   specified earlier using :ref:`InferRequest.setTensor <setTensor>` or
   :ref:`InferRequest.setInputTensor <setInputTensor>`

   * **Returns:**

     .. code-block:: ts

        {
        [outputName: string]: Tensor;
        }


   * **Defined in:**
     `addon.ts:460 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L460>`__


   .. code-block:: ts

      infer(inputData): {
          [outputName: string]: Tensor;
      }

   It infers specified input(s) in the synchronous mode.

   * **Parameters:**

     -

       .. code-block:: ts

          inputData: {
                     [inputName: string]: Tensor | SupportedTypedArray;
                     }

       An object with the key-value pairs where the key is the
       input name and value can be either a tensor or a ``TypedArray``. ``TypedArray``
       will be wrapped into ``Tensor`` underneath using the input shape and element type
       of the deployed model.

   * **Returns:**

     .. code-block:: ts

        {
        [outputName: string]: Tensor;
        }

   * **Defined in:**
     `addon.ts:468 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L468>`__


   .. code-block:: ts

      infer(inputData): {
          [outputName: string]: Tensor;
      }


   It infers specified input(s) in the synchronous mode.

   * **Parameters:**

     -

       .. code-block:: ts

          inputData: Tensor[] | SupportedTypedArray[]

       An array with tensors or ``TypedArrays``. ``TypedArrays`` will be
       wrapped into ``Tensors`` underneath using the input shape and element type
       of the deployed model. If the model has multiple inputs, the ``Tensors``
       and ``TypedArrays`` must be passed in the correct order.

   * **Returns:**

     .. code-block:: ts

        {
        [outputName: string]: Tensor;
        }

   * **Defined in:**
     `addon.ts:477 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L477>`__


.. rubric:: inferAsync

*

   .. code-block:: ts

      inferAsync(inputData): Promise<{
          [outputName: string]: Tensor;
      }>

   It infers specified input(s) in the asynchronous mode.

   * **Parameters:**

     -

       .. code-block:: ts

          inputData: Tensor[] | {
              [inputName: string]: Tensor;
          }

       An object with the key-value pairs where the key is the input name and
       value is a tensor or an array with tensors. If the model has multiple
       inputs, the Tensors must be passed in the correct order.

   * **Returns:**

     .. code-block:: ts

        Promise<{
         [outputName: string]: Tensor;
        }>


   * **Defined in:**
     `addon.ts:485 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L485>`__


.. rubric:: setInputTensor
   :name: setInputTensor

*

   .. code-block:: ts

      setInputTensor(tensor): void


   It sets the input tensor to infer models with a single input.

   * **Parameters:**

     - :doc:`Tensor <Tensor>`

       The input tensor. The element type and shape of the tensor must match
       the type and size of the model's input element. If the model has
       several inputs, an exception is thrown.

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:532 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L532>`__


   .. code-block:: ts

      setInputTensor(idx, tensor): void

   It sets the input tensor to infer.

   * **Parameters:**

     - idx: number

       The input tensor index. If idx is greater than the number of
       model inputs, an exception is thrown.

     - :doc:`Tensor <Tensor>`

       The input tensor. The element type and shape of the tensor
       must match the input element type and size of the model.

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:540 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L540>`__


.. rubric:: setOutputTensor

*

   .. code-block:: ts

      setOutputTensor(tensor): void

   It sets the output tensor to infer models with a single output.

   * **Parameters:**

     - :doc:`Tensor <Tensor>`

       The output tensor. The element type and shape of the tensor must match
       the output element type and size of the model. If the model has several
       outputs, an exception is thrown.

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:547 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L547>`__


   .. code-block:: ts

      setOutputTensor(idx, tensor): void

   It sets the output tensor to infer.

   * **Parameters:**

     - idx: number

       The output tensor index.

     - :doc:`Tensor <Tensor>`

       The output tensor. The element type and shape of the tensor
       must match the output element type and size of the model.

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:554 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L554>`__


.. rubric:: setTensor
   :name: setTensor

*

   .. code-block:: ts

      setTensor(name, tensor): void

   It sets the input/output tensor to infer.

   * **Parameters:**

     - name: string

       The input or output tensor name.

     - tensor: :doc:`Tensor <Tensor>`

       The tensor. The element type and shape of the tensor
       must match the input/output element type and size of the model.

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:561 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L561>`__

