# Asynchronous Compilation for Dynamic Models

## Motivation

When the input shape of any layer changes, a new static kernel must be compiled because the previously built kernel is no longer valid. If inference requests must wait for this compilation to complete, latency increases by the kernel compile time, degrading overall throughput. Asynchronous kernel compilation solves this problem by decoupling kernel compilation from inference execution.

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

The diagram above illustrates the overall async compilation flow. During network loading, dynamic kernels are selected and stored in the impl cache. At inference time, if the input shape of a primitive changes or its current implementation is dynamic, the runtime checks the impl cache for a pre-built implementation matching the new shape. On a cache hit, the pre-built implementation is used directly. On a cache miss, a new static kernel compilation task is dispatched in the background, and the dynamic kernel handles execution in the meantime to avoid stalling inference.

## Prioritized Asynchronous Kernel Compilation

Kernel compilation tasks for new input shapes run in threads separate from the inference thread. Without prioritization, less critical kernels could be compiled before performance-critical ones, reducing overall throughput. To address this, async compilation is restricted to four performance-critical primitives: convolution, fully-connected, GEMM, and softmax.
