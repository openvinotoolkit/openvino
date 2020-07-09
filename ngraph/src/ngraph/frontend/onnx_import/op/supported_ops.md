Generally nGraph doesn't support tensors of types:

* `string`,
* `complex64`,
* `complex128`.

Value in `()` _parenthesis_ indicates that this op was introduced since the specific 
ONNX Standard opset version. 
Values seperated by `-` _dash_ indicate the changes were made to that op definition 
in the ONNX Standard. If there were minor changes they are usually supported by single 
implementation, otherwise there are multiple versions, each appropriate for specific opset 
version range.
For example, with the schema represented below the operator `Abs` is supported in all 
opset versions starting from `1` to `6` and to the latest opset version.

## Supported Ops:

| Name | ONNX Opset supported | nGraph opset support | Comment |
|------|----------------------------|---------|-----|
| Abs | 1-6- | 0,1 |
| Acos | 7- | 0,1 |
| Acosh | 9- | 0, | Have to change to only v1 ops (NGONNX-1015)
| Add | 1-6-7- | 0,1 |
| And | 1-7- | 0,1 |
| ArgMax | 1- | 0,1 |
| ArgMin | 1- | 0,1 |
| Asin | 7- | 0,1 |
| Asinh | 9- | 0, | Have to change to only v1 ops (NGONNX-1015)
| Atan | 7 - | 0,1 |
| Atanh | 9- | 0, | Have to change to only v1 ops (NGONNX-1015)
| AveragePool | 1-7- | 0,1 |
| BatchNormalization | 1-6-7- | 0,1 |
| Cast | 1-6-9- | 0,1 |
| Ceil | 1-6- | 0,1 |
| Clip | 1-6- | 0,1 |
| Concat | 1-4- | 0,1 |
| Constant | 1- | 0,1 |
| Conv | 1- | 0,1 |
| ConvInteger | 10- | 0, |
| ConvTranspose | 1- | 0,1 |
| Cos | 7- | 0,1 |
| Cosh | 9- | 0,1 | Have to change to only v1 ops (NGONNX-1015)
| CumSum | 11- | 0, | NGONNX-944
| DepthToSpace | 1-11- | 0,1 |
| DequantizeLinear | 10- | 0, |
| Div | 1-6-7- | 0,1 | 
| Dropout | (1-6-7)-10- | 0,1 | Only for inference.
| Elu | 1-6- | 0,1 |
| Equal | 1-7 | 0,1 |
| Erf | 9- | 0,1 |
| Exp | 1-6- | 0,1 |
| Expand | 8- | 0,1 |  Only static version
| EyeLike | 9- | 0,1 |
| Flatten | 1-9- | 0,1 |
| Floor | 1-6- | 0,1 |
| Gather | 11- | 0,1 |
| GatherND | 11- | 0, |
| Gemm | 1-6-7-9-11 | 0, | Some tests failing (NGONNX-494), Have to change to only v1 ops (NGONNX-1015)
| GlobalAveragePool | 1- | 0,1 |
| GlobalLpPool | 1-2- | 0, (1) | Not fully v1, need `lp_norm` expressed with v1 ops (NGONNX-1018)
| GlobalMaxPool | 1- | 0,1 |
| Greater | 1-7-9 | 0,1 |
| HardSigmoid | 1-6- | 0,1 |
| Hardmax | 11- | 0, (1) | GOE
| Identity | 1- | 0,1 | 
| InstanceNormalization | 1- | 0, (1) | Have to change to only v1 ops (NGONNX-1015)
| LRN | 1- | 0,1 |
| LSTM | 1-7- | 0,(1) |
| LeakyRelu | 1-6- | 0,(1) | (NGONNX-1015)
| Less | 1-7-9 | 0,1 |
| Log | 1-6- | 0,1 |
| LogSoftmax | 1- | 0,1 |
| LpNormalization | 1- | 0, | (NGONNX-1018) need to update some builders
| MatMul | 1-9 | 0,(1) | Uses `v0::Dot`, v0 broadcasts and reshapes, update builders
| MatMulInteger | 10- | 0, | `v0::QuantizedDot`
| Max | 1-6-8- | 0, 1 |
| MaxPool | 1-8- | 0, 1 |
| Mean | 1-6-8- | 0, 1 |
| Min | 1-6-8- | 0,1 |
| Mod | 10- | 1 | 
| Mul | (1-6-)7- | 0,1 | 
| Neg | 1-6- | 0,1 |
| NonMaxSuppression | 11- | 1 |
| Not | 1- | 0,1 | (aka `v1::LogicalNot`)
| OneHot | (9) | 0, (1) | (NGONNX-1015)
| Or | 1-7- | 0,1 | (aka `v1::LogicalOr`)
| PRelu | 1-6-7-9 | 0, 1 |  fused op uses arithmetic and broadcasting from v0
| Pad | 1-2-11- | 0, (1) | (NGONNX-1015)
| Pow | 1-7- | 0,1 |
| QLinearConv | 10- | 0 | `opset0::QuantizedConvolution`
| QLinearMatMul | 10- | 0 | `v0::QuantizedDot`
| QuantizeLinear | 10- | 0 | `opset0::Quantize`
| Reciprocal | 1-6- | 0, 1| 
| ReduceL1 | 1- | 0, | (NGONNX-1018)
| ReduceL2 | 1- | 0,1 |
| ReduceLogSum | 1- | 0,1 |
| ReduceLogSumExp | 1- | 0,1 |
| ReduceMax | 1- | 0,1 | 
| ReduceMean | 1- | 0,1 |
| ReduceMin | 1- |  0,1 |
| ReduceProd | 1- | 0,1 |
| ReduceSum | 1- | 0,1 | 
| ReduceSumSquare | 1- | 0,1 |
| Relu | 1-6- | 0,1 |
| Reshape | 1-5- | (0,1) | v1 supports dynamic target shape, but only as Constant? 
| ReverseSequence | 10- | 0,1 |
| ScatterND | 11- | 0, | 
| Selu | 1-6- | 0, 1 |
| Shape | 1- | 0,1 |
| Shrink | 1- | 0,1 |
| Sigmoid | 1-6- | 0,1 |
| Sign | 9- | 0,1 |
| Sin | 7- | 0,1 |
| Sinh | 9- | 0,1 |
| Size | 1- | 0,1 |
| Slice | 1- | 0,1 |
| Softmax | 1- | 0,1 |
| Softplus | 1- | 0,1 |
| Softsign | 1- | 0,(1) | (NGONNX-1015)
| SpaceToDepth | 1- | 0,1 | 
| Split | 1-2- | 0,1 |
| Sqrt | 1-6- | 0,1 |
| Squeeze | 1- | 0,(1) | 
| Sub | (1-6-)7- | 0,1 |
| Sum | 1-6-8- | 0,1 | 
| Tan | 7- | 0,1 |
| Tanh | 1-6- | 0,1 |
| ThresholdedRelu | 10- | 0,1 |
| TopK | 1- | 0,(1) | Need v0::GOE
| Transpose | 1- | 0,1 |
| Unsqueeze | 1- | 0,1 |
| Xor | 1-7- | 0,1 |
| Where | 9- | 0,1 |

### Able to implement or WIP
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| BitShift | (11)- |  | 1014 |
| ConstantOfShape | (9) | 286 | 445 | Dynamic shape input. WIP |
| DynamicQuantizeLinear | (11) | | 786 |
| GRU | - | | 325, 177 | There is no `GRUCell` nor `GRU` in v1 |
| RNN | - | | 323, 287 | `v1::RNNCell`|
| Round | (11)- | | 1008 | `v0::Round`
| Tile | - | NGRAPH-3292 | 368 | Dynamic op. WIP |
| Cast | 1-6- | 290 | 452 | Float16 unsupported. |

## Unsupported Ops:

### Lack of support in nGraph
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| MaxUnpool | (9) | 286, 289 | 447 | |
| LpPool | - | 291 | 488 | Unsupported by nGraph - only max/avg pooling ops. Need separate kernel. |
| Multinomial | - | 199 | 435 | Lack of PRNG in nGraph. |
| RandomNormal | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomNormalLike | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomUniform | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomUniformLike | - | 199 | 434 | Lack of PRNG in nGraph. |
| IsInf | (10) | | 528 | 
| StringNormalizer | (10) | | 600 | Need support for `string` data type.
| TfIdfVectorizer | (9) | | 523 |
| Det | (11) | | 754 | |

### Futher analysis needed
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| If | - | | 432 | At this moment probably impossible. |
| IsNaN | (9) | | 440 | Hacky way is to generate constant nodes with representations of NaN and compare with them. |
| Loop | - | | 432 | Static loops with some preconditions may be possible, however no idea how to pass graph (proto?) as a _body_ attribute. (what about graph contains `Loop`?) |
| Scan | - |  | 433 | Further analysis needed. - determine whether it is possible to import graph passed by op attribute. |
| Einsum | (12) | | | User can define in a language the operation to perform |
| NonZero | (9) | | 472 | Maybe we can leverage TopK here? First count NonZero elements with logic ops and reduction and then TopK. 
| Resize | (10-11)- | | 782 | Look like Interpolation over ROIs. Very specialized types of interpolation.
| ScatterElements | (11) |  | 977 | 
| ScatterND | (11) |  | 1020 | 
| Unique | (11) | | 761 |

### Dynamic operators
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| Compress | (9-11) | 285 | 438 | Dynamically selected indices |
| Expand | - | NGRAPH-3289 | 367 | Dynamic op. |
| GatherElements | - |  | 757 |  |
| OneHot | (9) | NGCORE-339 | 486 | Dynamic output shape
| Upsample | (7-9-10-) | 287 | 441 | Dynamic op. **Deprecated** from opset 10 |
| MaxRoiPool | - | 288 | 487 | Dynamic op - Need dynamic slicing. Beside just use _slice/op/concat_ pattern. |
| Reshape | 1-5- | NGRAPH-3290 | 357 | Lack of support for dynamic shape input. Only as a Constant or as an Initializer. |
| Scatter | (9) | 289 | 446 | Dynamic indices input. **Deprecated** in ONNX standard |
| RoiAlign | (10) | | 601 | 

### Sequence* ops
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| ConcatFromSequence | (11)- |  | 1016 |
| SequenceAt | (11) | | 1021 | need further analysis |
| SequenceConstruct | (11) | | 1021 | need further analysis |
| SequenceEmpty | (11) | | 1021 | need further analysis |
| SequenceErase | (11) | | 1021 | need further analysis |
| SequenceInsert | (11) | | 1021 | need further analysis |
| SequenceLength | (11) | | 1021 | need further analysis |
| SplitToSequence | (11) | | 1021 | need further analysis |
