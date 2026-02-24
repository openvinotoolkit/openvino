Interface AsyncInferQueue
========================

.. code-block:: ts

   interface AsyncInferQueue {
       new (compiledModel: CompiledModel, jobs?: number): AsyncInferQueue;
       setCallback(callback: AsyncInferQueueCallback): void;
       startAsync(inputData, userData?): Promise<object>;
       release(): void;
   }

``AsyncInferQueue`` manages a pool of :doc:`InferRequest <InferRequest>`
objects for asynchronous inference. It automatically spawns the pool and
provides synchronization to control the pipeline flow.

* **Defined in:**
  `addon.ts:725 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L725>`__


Constructor
#####################

.. code-block:: ts

   new AsyncInferQueue(compiledModel: CompiledModel, jobs?: number): AsyncInferQueue

Creates an ``AsyncInferQueue``.

* **Parameters:**

  - compiledModel: :doc:`CompiledModel <CompiledModel>`

    The compiled model used to create ``InferRequest`` objects in the pool.

  - ``Optional``

    .. code-block:: ts

       jobs: number

    Number of ``InferRequest`` objects in the pool. If not provided, the
    optimal number is set automatically.

* **Returns:** AsyncInferQueue

* **Defined in:**
  `addon.ts:733 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L733>`__


Type Aliases
#####################

.. rubric:: AsyncInferQueueCallback

.. code-block:: ts

   type AsyncInferQueueCallback = (
     error: null | Error,
     inferRequest: InferRequest,
     userData: object,
   ) => void;

Callback function type for ``AsyncInferQueue`` operations.

* **Parameters:**

  - error: ``null | Error`` — Error that occurred during inference, if any.
  - inferRequest: :doc:`InferRequest <InferRequest>` — The request from the
    pool, providing access to input and output tensors.
  - userData: ``object`` — User data passed to ``startAsync``. If not
    provided, it will be ``undefined``.

* **Defined in:**
  `addon.ts:719 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L719>`__


Methods
#####################


.. rubric:: setCallback

*

   .. code-block:: ts

      setCallback(callback: AsyncInferQueueCallback): void

   Sets a unified callback on all ``InferRequest`` objects from the pool.
   Any previously set callback is replaced.

   * **Parameters:**

     - callback: AsyncInferQueueCallback

       A function matching the callback requirements (see type alias above).

   * **Returns:** void

   * **Defined in:**
     `addon.ts:739 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L739>`__


.. rubric:: startAsync

*

   .. code-block:: ts

      startAsync(
        inputData: { [inputName: string]: Tensor } | Tensor[],
        userData?: object,
      ): Promise<object>

   Starts asynchronous inference for the specified input data. If all
   requests in the pool are busy, waits for one to become available.

   * **Parameters:**

     - inputData: ``{ [inputName: string]: Tensor } | Tensor[]``

       An object with key-value pairs where the key is the input name
       and value is a :doc:`Tensor <Tensor>`, or an array of tensors.

     - ``Optional``

       .. code-block:: ts

          userData: object

       User data that will be passed to the callback.

   * **Returns:** ``Promise<object>``

     A Promise that resolves when the inference callback completes.

   * **Defined in:**
     `addon.ts:745 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L745>`__


.. rubric:: release

*

   .. code-block:: ts

      release(): void

   Releases resources associated with this ``AsyncInferQueue`` instance.
   Call this method after all ``startAsync`` requests have completed
   and the ``AsyncInferQueue`` is no longer needed.

   * **Returns:** void

   * **Defined in:**
     `addon.ts:756 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L756>`__
