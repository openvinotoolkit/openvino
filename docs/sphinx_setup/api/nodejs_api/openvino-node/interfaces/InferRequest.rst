InferRequest
============

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

* **Defined in:**
  `addon.ts:101 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L101>`__


Methods
#####################


.. rubric:: getCompiledModel

.. container:: m-4

   .. code-block:: ts

      getCompiledModel(): CompiledModel

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:112 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L112>`__


.. rubric:: getInputTensor

.. container:: m-4

   .. code-block:: ts

      getInputTensor(idx?): Tensor


   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          idx: number


   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:106 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L106>`__


.. rubric:: getOutputTensor

.. container:: m-4

   .. code-block:: ts

      getOutputTensor(idx?): Tensor


   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          idx: number

   * **Returns:**  :doc:`Tensor <Tensor>`


   * **Defined in:**
     `addon.ts:107 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L107>`__

.. rubric:: getTensor

.. container:: m-4

   .. code-block:: ts

      getTensor(nameOrOutput): Tensor

   * **Parameters:**

     - nameOrOutput: string| :doc:`Output <Output>`

   * **Returns:**  :doc:`Tensor <Tensor>`

   * **Defined in:**
     `addon.ts:105 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L105>`__


.. rubric:: infer

.. container:: m-4

   .. code-block:: ts

      infer(inputData?): {
          [outputName: string]: Tensor;
      }

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          inputData: {
                     [inputName: string]: Tensor | SupportedTypedArray;
                     } | Tensor[] | SupportedTypedArray[]

   * **Returns:**

     .. code-block:: ts

        {
        [outputName: string]: Tensor;
        }

     - [outputName: string]: Tensor

   * **Defined in:**
     `addon.ts:108 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L108>`__


.. rubric:: inferAsync

.. container:: m-4

   .. code-block:: ts

      inferAsync(inputData): Promise<{
          [outputName: string]: Tensor;
      }>

   * **Parameters:**

     -

       .. code-block:: ts

          inputData: Tensor[] | {
              [inputName: string]: Tensor;
          }

   * **Returns:**

     .. code-block:: ts

        Promise<{
         [outputName: string]: Tensor;
        }>


   * **Defined in:**
     `addon.ts:110 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L110>`__

.. rubric:: setInputTensor

.. container:: m-4

   .. code-block:: ts

      setInputTensor(idxOrTensor, tensor?): void


   * **Parameters:**

     - idxOrTensor: number| :doc:`Tensor <Tensor>`

     - ``Optional``

       .. code-block:: ts

          tensor: Tensor


   * **Returns:**  void

   * **Defined in:**
     `addon.ts:103 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L103>`__

.. rubric:: setOutputTensor

.. container:: m-4

   .. code-block:: ts

      setOutputTensor(idxOrTensor, tensor?): void


   * **Parameters:**

     - idxOrTensor: number| :doc:`Tensor <Tensor>`
     - ``Optional``

       .. code-block:: ts

          tensor: Tensor


   * **Returns:**  void

   * **Defined in:**
     `addon.ts:104 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L104>`__

.. rubric:: setTensor

.. container:: m-4

   .. code-block:: ts

      setTensor(name, tensor): void

   * **Parameters:**

     - name: string
     - tensor: :doc:`Tensor <Tensor>`

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:102 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L102>`__

