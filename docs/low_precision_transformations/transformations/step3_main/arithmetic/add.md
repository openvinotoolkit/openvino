# OpenVINO™ Low Precision Transformations: AddTransformation {#openvino_docs_IE_DG_lpt_AddTransformation}

ngraph::pass::low_precision::AddTransformation class represents the `Add` operation transformation.

The transformation propagates dequantization subtraction from one input branch to another and propagates dequantization multiplication from the same branch through `Add` operation. In transformation result, one `Add` operation input branch is in low precision without dequantization operations (empty branch), another input branch is in original precision with updated dequantization operations (full branch).

Empty branch selection criteria step by step (by priority):

*Step #1.* If one branch is quantized only, then the quantized branch is empty branch. 

*Step #2.* If only one branch has `FakeQuantize` before dequantization operations, then another branch is empty branch. 

*Step #3.* If some `FakeQuantize` has more then one consumers and another only one, then the branch with `FakeQuantize` with several consumers is empty branch. 

*Step #4.* Constant branch is in original precision, data branch is empty branch. In this case dequantization operations are propagated to constant branch and will be fused in one constant.

*Step #5.* If both branches before `FakeQuantize` have operations from the list: `Convolution`, `GroupConvolution` and `MatMul`, or both branches before `FakeQuantize` don't have any operations from the same list, then the branch with larger shape volume is empty branch.

*Step #6.* If in some branch an operation before `FakeQuantize` has several consumers then the branch is empty.

If dequantization operations on the full branch have `FakeQuantize` operation parent, then they will be fused with `FakeQuantize` during another low precision transformations. If `FakeQuantize` operation has a parent operation from the list: `Convolution`, `GroupConvolution` and `MatMul`, then during inference the `FakeQuantize` can be inferred in one plugin kernel with the parent operation.

Depending on the plugin instruction set, low precision inference for `Add` operation can be implemented in two logical steps in one plugin kernel:

 * Inference step #1: Operations in the full branch (for example: `Convolution` and `FakeQuantize` with fused dequantization operations) and `Add` can be inferred in original precision.

 * Inference step #2: Inference step #1 result can be added with empty branch tensor in low precision.

This approach allows to infer `Add` operation in the most optimal way.

## Subgraph before transformation
The subgraph with quantized `Add` operation before transformation:

\f[
y_{ch,i}=(scale1_{ch} * (x1_{ch,i} - shift1_{ch})) + (scale2_{ch} * (x2_{ch,i} - shift2_{ch}))
\f]

![Add before](img/add.common.png)

## Subgraph after transformation
The subgraph with `Add` operation after the transformation:

\f[
y_{ch,i}=scale2_{ch} * (scale1_{ch}' * (x1_{ch,i} - shift1_{ch}') + x2_{ch,i})
\f]

where:

\f[
scale1_{ch}' = scale1_{ch} / scale2_{ch}
\f]

\f[
shift1_{ch}' = shift1_{ch} + scale2_{ch} * shift2_{ch} / scale1_{ch}
\f]

![Add before](img/add.transformed.png)