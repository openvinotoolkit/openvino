# Internal CPU Plugin Optimizations

The CPU plugin supports several graph optimization algorithms, such as fusing or removing layers.
Refer to the sections below for details.

> **NOTE**: For layer descriptions, see the [IR Notation Reference](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/available-opsets.html).


## Fusing Convolution and Simple Layers

Merge of a convolution layer and any of the simple layers listed below:
- Activation: ReLU, ELU, Sigmoid, Clamp
- Depthwise: ScaleShift, PReLU
- FakeQuantize

> **NOTE**: You can have any number and order of simple layers.

A combination of a convolution layer and simple layers results in a single fused layer called
*Convolution*:

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA2(Convolution)
    nodeA2(Convolution) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input) --> nodeB2(Convolution)
    nodeB2(Convolution) --> nodeB3(Simple Layer)
    nodeB3(Simple Layer) --> nodeB4(...)
    nodeB4(...) --> nodeB5(Simple Layer)
    nodeB5(Simple Layer) --> nodeB6(Output)
    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
class subgraphA1,subgraphB1,nodeB4 no-bg-color
class nodeA2 daisy1
class nodeB1,nodeB6,nodeA1,nodeA3 moss1
class nodeB2,nodeB3,nodeB5, steel1
```

## Fusing Pooling and FakeQuantize Layers

A combination of Pooling and FakeQuantize layers results in a single fused layer called *Pooling*:

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA2(Pooling)
    nodeA2(Pooling) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input) --> nodeB2("Pooling [Average]")
    nodeB2("Pooling [Average]") --> nodeB3(Fake Quantize)
    nodeB3(Fake Quantize) --> nodeB4(Output)
    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
class subgraphA1,subgraphB1 no-bg-color
class nodeA2 daisy1
class nodeB1,nodeB4,nodeA1,nodeA3 moss1
class nodeB2,nodeB3 steel1
```
## Fusing FullyConnected and Activation Layers

A combination of FullyConnected and Activation layers results in a single fused layer called
*FullyConnected*:

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA2(FullyConnected)
    nodeA2(FullyConnected) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input) --> nodeB2(FullyConnected)
    nodeB2(FullyConnected) --> nodeB3("Activation [ReLU]")
    nodeB3("Activation [ReLU]") --> nodeB4(Output)
    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
class subgraphA1,subgraphB1 no-bg-color
class nodeA2 daisy1
class nodeB1,nodeB4,nodeA1,nodeA3 moss1
class nodeB2,nodeB3 steel1
```
## Fusing Convolution and Depthwise Convolution Layers Grouped with Simple Layers

> **NOTE**: This pattern is possible only on CPUs with support of Streaming SIMD Extensions 4.2
> (SSE 4.2) and Intel AVX2 Instruction Set Architecture (ISA).

A combination of a group of a Convolution (or Binary Convolution) layer and simple layers and a group of a Depthwise Convolution
layer and simple layers results in a single layer called *Convolution* (or *Binary Convolution*):
> **NOTE**: Depthwise convolution layers should have the same values for the `group`, input channels, and output channels parameters.

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA2(Convolution)
    nodeA2(Convolution) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input) --> nodeB2(Convolution)
    nodeB2(Convolution) --> nodeB3(Simple Layer)
    nodeB3(Simple Layer) --> nodeB4(...)
    nodeB4(...) --> nodeB5(Simple Layer)
    nodeB5(Simple Layer) --> nodeB6(Depthwise \n Convolution)
    nodeB6(Depthwise \n Convolution) --> nodeB7(Simple Layer)
    nodeB7(Simple Layer) --> nodeB8(...)
    nodeB8(...) --> nodeB9(Simple Layer)
    nodeB9(Simple Layer) --> nodeB10(Output)
    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
class subgraphA1,subgraphB1,nodeB4,nodeB8 no-bg-color
class nodeA2 daisy1
class nodeB1,nodeA1,nodeA3,nodeB10 moss1
class nodeB2,nodeB3,nodeB5,nodeB6,nodeB7,nodeB9 steel1
```
## Fusing Convolution and Sum Layers

A combination of convolution, simple, and Eltwise layers with the sum operation results in a single layer called *Convolution*:

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA4(Any Layer)
    nodeA4(Any Layer) --> nodeA2(Convolution)
    nodeA5(Input2) ---> nodeA2(Convolution)
    nodeA2(Convolution) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input1) --> nodeB7(Any Layer)
    nodeB7(Any Layer) -----> nodeB2("Eltwise[op=sum]")
    nodeB8(Input) --> nodeB9(Convolution)
    nodeB9(Convolution) --> nodeB10(Simple Layer)
    nodeB10(Simple Layer) --> nodeB11(...)
    nodeB11(...) --> nodeB12(Simple Layer)
    nodeB12(Simple Layer) --> nodeB2("Eltwise[op=sum]")
    nodeB2("Eltwise[op=sum]") --> nodeB3(Simple Layer)
    nodeB3(Simple Layer) --> nodeB4(...)
    nodeB4(...) --> nodeB5(Simple Layer)
    nodeB5(Simple Layer) --> nodeB6(Output)
    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
classDef coral1 fill:#FFB6B9, stroke: #FF848A, color: #262626
class subgraphA1,subgraphB1,nodeB4,nodeB11 no-bg-color
class nodeA2 daisy1
class nodeB1,nodeA5,nodeA1,nodeA3,nodeB6,nodeB8 moss1
class nodeB3,nodeB5,nodeA4,nodeB7,nodeB9,nodeB10,nodeB12 steel1
class nodeB2 coral1
```
## Fusing a Group of Convolutions

If a topology contains the following pipeline, a CPU plugin merges split, convolution, and concatenation layers into a single convolution layer with the group parameter:

```mermaid
flowchart TD
    subgraph subgraphA1[Runtime Graph]
    direction TB
    nodeA1(Input) --> nodeA2(Convolution)
    nodeA2(Convolution) --> nodeA3(Output)
    end
    subgraph subgraphB1[Original Graph]
    direction TB
    nodeB1(Input) --> nodeB2(Split)
    nodeB2(Split) --> nodeB6(Convolution1)
    nodeB6(Convolution1) --> nodeB4(Concatenation)
    nodeB2(Split) --> nodeB3(Convolution3)
    nodeB2(Split) --> nodeB7(Convolution2)
    nodeB7(Convolution2) --> nodeB4(Concatenation)
    nodeB3(Convolution3) --> nodeB4(Concatenation)
    nodeB4(Concatenation) --> nodeB5(Output)

    end
classDef no-bg-color fill:none,stroke-width:0px
classDef moss1 fill:#D7F3A2, stroke: #B1D272, color: #262626
classDef steel1 fill:#B9D6E5, stroke: #86B3CA, color: #262626
classDef daisy1 fill:#FFE17A, stroke: #FEC91B, color: #262626
classDef coral-tint-2 fill:#FFB6B9, stroke: #FF848A, color: #262626
class subgraphA1,subgraphB1 no-bg-color
class nodeB4,nodeB2 coral-tint-2
class nodeA2 daisy1
class nodeB1,nodeA1,nodeA3,nodeB5 moss1
class nodeB3,nodeB6,nodeB7 steel1
```
> **NOTE**: Parameters of the convolution layers must coincide.


## Removing a Power Layer

CPU plugin removes a Power layer from a topology if it has the following parameters:
  - <b>power</b> = 1
  - <b>scale</b> = 1
  - <b>offset</b> = 0

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)