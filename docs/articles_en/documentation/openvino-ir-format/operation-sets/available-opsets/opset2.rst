opset2
======


.. meta::
  :description: Explore the examples of operation instances expressed as IR
                XML snippets in the opset2 operation set, supported in OpenVINO™
                toolkit.

This specification document describes ``opset2`` operation set supported in OpenVINO™.
Support for each particular operation from the list below depends on the capabilities of an inference plugin
and may vary among different hardware platforms and devices. Examples of operation instances are provided as IR xml
snippets. The semantics match corresponding OpenVINO operation classes declared in ``namespace opset2``.


Table of Contents
######################

* :doc:`Abs <../operation-specs/arithmetic/abs-1>`
* :doc:`Acos <../operation-specs/arithmetic/acos-1>`
* :doc:`Add <../operation-specs/arithmetic/add-1>`
* :doc:`Asin <../operation-specs/arithmetic/asin-1>`
* :doc:`Atan <../operation-specs/arithmetic/atan-1>`
* :doc:`AvgPool <../operation-specs/pooling/avg-pool-1>`
* :doc:`BatchNormInference <../operation-specs/normalization/batch-norm-inference-1>`
* :doc:`BatchToSpace <../operation-specs/movement/batch-to-space-2>`
* :doc:`BinaryConvolution <../operation-specs/convolution/binary-convolution-1>`
* :doc:`Broadcast <../operation-specs/movement/broadcast-1>`
* :doc:`CTCGreedyDecoder <../operation-specs/sequence/ctc-greedy-decoder-1>`
* :doc:`Ceiling <../operation-specs/arithmetic/ceiling-1>`
* :doc:`Clamp <../operation-specs/activation/clamp-1>`
* :doc:`Concat <../operation-specs/movement/concat-1>`
* :doc:`Constant <../operation-specs/infrastructure/constant-1>`
* :doc:`Convert <../operation-specs/type/convert-1>`
* :doc:`ConvertLike <../operation-specs/type/convert-like-1>`
* :doc:`Convolution <../operation-specs/convolution/convolution-1>`
* :doc:`ConvolutionBackpropData <../operation-specs/convolution/convolution-backprop-data-1>`
* :doc:`Cos <../operation-specs/arithmetic/cos-1>`
* :doc:`Cosh <../operation-specs/arithmetic/cosh-1>`
* :doc:`DeformableConvolution <../operation-specs/convolution/deformable-convolution-1>`
* :doc:`DeformablePSROIPooling <../operation-specs/detection/deformable-psroi-pooling-1>`
* :doc:`DepthToSpace <../operation-specs/movement/depth-to-space-1>`
* :doc:`DetectionOutput <../operation-specs/detection/detectionoutput-1>`
* :doc:`Divide <../operation-specs/arithmetic/divide-1>`
* :doc:`Elu <../operation-specs/activation/elu-1>`
* :doc:`Equal <../operation-specs/comparison/equal-1>`
* :doc:`Erf <../operation-specs/arithmetic/erf-1>`
* :doc:`Exp <../operation-specs/activation/exp-1>`
* :doc:`FakeQuantize <../operation-specs/quantization/fake-quantize-1>`
* :doc:`Floor <../operation-specs/arithmetic/floor-1>`
* :doc:`FloorMod <../operation-specs/arithmetic/floormod-1>`
* :doc:`Gather <../operation-specs/movement/gather-1>`
* :doc:`GatherTree <../operation-specs/movement/gather-tree-1>`
* :doc:`Gelu <../operation-specs/activation/gelu-2>`
* :doc:`Greater <../operation-specs/comparison/greater-1>`
* :doc:`GreaterEqual <../operation-specs/comparison/greater-equal-1>`
* :doc:`GRN <../operation-specs/normalization/grn-1>`
* :doc:`GroupConvolution <../operation-specs/convolution/group-convolution-1>`
* :doc:`GroupConvolutionBackpropData <../operation-specs/convolution/group-convolution-backprop-data-1>`
* :doc:`HardSigmoid <../operation-specs/activation/hard-sigmoid-1>`
* :doc:`Interpolate <../operation-specs/image/interpolate-1>`
* :doc:`Less <../operation-specs/comparison/less-1>`
* :doc:`LessEqual <../operation-specs/comparison/lessequal-1>`
* :doc:`Log <../operation-specs/arithmetic/log-1>`
* :doc:`LogicalAnd <../operation-specs/logical/logical-and-1>`
* :doc:`LogicalNot <../operation-specs/logical/logical-not-1>`
* :doc:`LogicalOr <../operation-specs/logical/logical-or-1>`
* :doc:`LogicalXor <../operation-specs/logical/logical-xor-1>`
* :doc:`LRN <../operation-specs/normalization/lrn-1>`
* :doc:`LSTMCell <../operation-specs/sequence/lstm-cell-1>`
* :doc:`MatMul <../operation-specs/matrix/matmul-1>`
* :doc:`MaxPool <../operation-specs/pooling/max-pool-1>`
* :doc:`Maximum <../operation-specs/arithmetic/maximum-1>`
* :doc:`Minimum <../operation-specs/arithmetic/minimum-1>`
* :doc:`Mod <../operation-specs/arithmetic/mod-1>`
* :doc:`MVN <../operation-specs/normalization/mvn-1>`
* :doc:`Multiply <../operation-specs/arithmetic/multiply-1>`
* :doc:`Negative <../operation-specs/arithmetic/negative-1>`
* :doc:`NonMaxSuppression <../operation-specs/sort/non-max-suppression-1>`
* :doc:`NormalizeL2 <../operation-specs/normalization/normalize-l2-1>`
* :doc:`NotEqual <../operation-specs/comparison/notequal-1>`
* :doc:`OneHot <../operation-specs/sequence/one-hot-1>`
* :doc:`Pad <../operation-specs/movement/pad-1>`
* :doc:`Parameter <../operation-specs/infrastructure/parameter-1>`
* :doc:`Power <../operation-specs/arithmetic/power-1>`
* :doc:`PReLU <../operation-specs/activation/prelu-1>`
* :doc:`PriorBoxClustered <../operation-specs/detection/prior-box-clustered-1>`
* :doc:`PriorBox <../operation-specs/detection/prior-box-1>`
* :doc:`Proposal <../operation-specs/detection/proposal-1>`
* :doc:`PSROIPooling <../operation-specs/detection/psroi-pooling-1>`
* :doc:`Range <../operation-specs/generation/range-1>`
* :doc:`ReLU <../operation-specs/activation/relu-1>`
* :doc:`ReduceLogicalAnd <../operation-specs/reduction/reduce-logical-and-1>`
* :doc:`ReduceLogicalOr <../operation-specs/reduction/reduce-logical-or-1>`
* :doc:`ReduceMax <../operation-specs/reduction/reduce-max-1>`
* :doc:`ReduceMean <../operation-specs/reduction/reduce-mean-1>`
* :doc:`ReduceMin <../operation-specs/reduction/reduce-min-1>`
* :doc:`ReduceProd <../operation-specs/reduction/reduce-prod-1>`
* :doc:`ReduceSum <../operation-specs/reduction/reduce-sum-1>`
* :doc:`RegionYolo <../operation-specs/detection/region-yolo-1>`
* :doc:`ReorgYolo <../operation-specs/detection/reorg-yolo-1>`
* :doc:`Reshape <../operation-specs/shape/reshape-1>`
* :doc:`Result <../operation-specs/infrastructure/result-1>`
* :doc:`ReverseSequence <../operation-specs/movement/reverse-sequence-1>`
* :doc:`ROIPooling <../operation-specs/detection/roi-pooling-1>`
* :doc:`Select <../operation-specs/condition/select-1>`
* :doc:`Selu <../operation-specs/activation/selu-1>`
* :doc:`ShapeOf <../operation-specs/shape/shape-of-1>`
* :doc:`Sigmoid <../operation-specs/activation/sigmoid-1>`
* :doc:`Sign <../operation-specs/arithmetic/sign-1>`
* :doc:`Sin <../operation-specs/arithmetic/sin-1>`
* :doc:`Sinh <../operation-specs/arithmetic/sinh-1>`
* :doc:`SoftMax <../operation-specs/activation/softmax-1>`
* :doc:`SpaceToBatch <../operation-specs/movement/space-to-batch-2>`
* :doc:`SpaceToDepth <../operation-specs/movement/space-to-depth-1>`
* :doc:`Split <../operation-specs/movement/split-1>`
* :doc:`Sqrt <../operation-specs/arithmetic/sqrt-1>`
* :doc:`SquaredDifference <../operation-specs/arithmetic/squared-difference-1>`
* :doc:`Squeeze <../operation-specs/shape/squeeze-1>`
* :doc:`StridedSlice <../operation-specs/movement/strided-slice-1>`
* :doc:`Subtract <../operation-specs/arithmetic/subtract-1>`
* :doc:`Tan <../operation-specs/arithmetic/tan-1>`
* :doc:`Tanh <../operation-specs/arithmetic/tanh-1>`
* :doc:`TensorIterator <../operation-specs/infrastructure/tensor-iterator-1>`
* :doc:`Tile <../operation-specs/movement/tile-1>`
* :doc:`TopK <../operation-specs/sort/top-k-1>`
* :doc:`Transpose <../operation-specs/movement/transpose-1>`
* :doc:`Unsqueeze <../operation-specs/shape/unsqueeze-1>`
* :doc:`VariadicSplit <../operation-specs/movement/variadic-split-1>`
