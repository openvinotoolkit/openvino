About nGraph Compiler stack
===========================

## Bridge

Starting from the top of the stack, nGraph receives a computational graph
from a deep learning framework such as TensorFlow or MXNet. The
computational graph is converted to an nGraph internal representation 
by a bridge created for the corresponding framework.

An nGraph bridge examines the whole graph to pattern match subgraphs
which nGraph knows how to execute, and these subgraphs are encapsulated.
Parts of the graph that are not encapsulated will default to framework
implementation when executed.


## nGraph Core

nGraph uses a strongly-typed and platform-neutral
`Intermediate Representation (IR)` to construct a "stateless"
computational graph. Each node, or op, in the graph corresponds to
one `step` in a computation, where each step produces zero or
more tensor outputs from zero or more tensor inputs.

This allows nGraph to apply its state of the art optimizations instead
of having to follow how a particular framework implements op execution,
memory management, data layouts, etc.

In addition, using nGraph IR allows faster optimization delivery
for many of the supported frameworks. For example, if nGraph optimizes
ResNet for TensorFlow, the same optimization can be readily applied
to MXNet* or ONNX* implementations of ResNet.


Features
--------

nGraph performs a combination of device-specific and
non-device-specific optimizations:

-   **Fusion** -- Fuse multiple ops to to decrease memory usage.
-   **Data layout abstraction** -- Make abstraction easier and faster
    with nGraph translating element order to work best for a given or
    available device.
-   **Data reuse** -- Save results and reuse for subgraphs with the
    same input.
