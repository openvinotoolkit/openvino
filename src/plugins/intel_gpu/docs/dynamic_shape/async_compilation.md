# Asynchronous Compilation for Dynamic Models

## Motivation

If the input shape of any layer changes, a new static kernel must be compiled because the previously built static kernel cannot be used. In this case, if inference requests need to wait for the compilation of the newly selected kernel to finish, the inference latency will increase by the compile time of the new kernel, causing the performance to decrease. To solve this problem, the asynchronous kernel compilation technique is introduced.

## Overall Workflow

<!-- flowchart TD
    A[Start Network Loading] -- > B(Build dynamic kernels)
    B -- > C[Start Inferencing]
    C -- > D{Is Input Shape Changed?
            or Is current impl dynamic?}
    D -- > |Yes| G{Does this primitive have a cached impl?}
    G -- > |Yes| I(Load pre-built impl from impl cache)
    I -- > F
    G -- > |No| H(Trigger a new kernel compilation task
                 Load dynamic kernel from the impl cache)
    D -- > |No| F(Execution)
    H -- > F -->

<img src="async_compilation.PNG" alt="async compilation overall workflow" width=500>

The above diagram shows the overall execution flow of the async compilation. First, in the case of network loading, dynamic kernels are selected and stored in the impl cache. Next, in the phase of network inference, if the input shape of the current primitive impl is changed or it has a dynamic kernel, it is checked whether the impl cache has a pre-built impl for the updated input shape. In the case of cache hit, the pre-built impl is used for the primitive execution. In the case of cache miss, a new static kernel compilation task is triggered in the backgroud, and the dynamic kernel is used at this time for faster inference.

## Prioritized Asynchronous Kernel Compilation

As explained before, kernel compilation tasks for new input shapes run in separate threads from network inference. In this case, the overall performance could be decreased if performance critical kernels are built later than less critical kernels. For this reason, async compilation are done for only four critical primitives, such as convolution, fully-connected, gemm and softmax.