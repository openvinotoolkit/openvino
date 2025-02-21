# ONNX Operations Supported by OpenVINO ONNX Frontend

Here is a table of operations available by the ONNX Frontend.

OpenVINO provides support for operations of Default Opset (empty in table below), other operations are supported by community.

| Domain                 | Operation Name                                         | Supported Version      | Defined Version                | Limitation                     |
|------------------------|--------------------------------------------------------|------------------------|--------------------------------|--------------------------------|
|                        |Abs                                                     |                        |13, 6, 1                        |                                |
|                        |Acos                                                    |                        |22, 7                           |                                |
|                        |Acosh                                                   |                        |22, 9                           |                                |
|                        |Add                                                     |                        |14, 13, 7, 6, 1                 |                                |
|                        |Affine                                                  |                        |                                |                                |
|                        |And                                                     |                        |7, 1                            |                                |
|                        |ArgMax                                                  |                        |13, 12, 11, 1                   |                                |
|                        |ArgMin                                                  |                        |13, 12, 11, 1                   |                                |
|                        |Asin                                                    |                        |22, 7                           |                                |
|                        |Asinh                                                   |                        |22, 9                           |                                |
|                        |Atan                                                    |                        |22, 7                           |                                |
|                        |Atanh                                                   |                        |22, 9                           |                                |
|                        |ATen                                                    |                        |                                |                                |
|                        |AveragePool                                             |                        |22, 19, 11, 10, 7, 1            |                                |
|                        |BatchNormalization                                      |                        |15, 14, 9, 7, 6, 1              |                                |
|                        |BitShift                                                |                        |11                              |                                |
|                        |BitwiseAnd                                              |                        |18                              |                                |
|                        |BitwiseNot                                              |                        |18                              |                                |
|                        |BitwiseOr                                               |                        |18                              |                                |
|                        |BitwiseXor                                              |                        |18                              |                                |
|                        |Cast                                                    |                        |21, 19, 13, 9, 6, 1             |                                |
|                        |Ceil                                                    |                        |13, 6, 1                        |                                |
|                        |Col2Im                                                  |                        |18                              |                                |
|                        |Compress                                                |                        |11, 9                           |                                |
|                        |Concat                                                  |                        |13, 11, 4, 1                    |                                |
|                        |ConcatFromSequence                                      |                        |11                              |                                |
|                        |Constant                                                |                        |21, 19, 13, 12, 11, 9, 1        |                                |
|                        |ConstantFill                                            |                        |                                |                                |
|                        |ConstantOfShape                                         |                        |21, 20, 9                       |                                |
|                        |Conv                                                    |                        |22, 11, 1                       |                                |
|                        |ConvInteger                                             |                        |10                              |                                |
|                        |ConvTranspose                                           |                        |22, 11, 1                       |                                |
|                        |Cos                                                     |                        |22, 7                           |                                |
|                        |Cosh                                                    |                        |22, 9                           |                                |
|                        |Crop                                                    |                        |                                |                                |
|                        |CumSum                                                  |                        |14, 11                          |                                |
|                        |DFT                                                     |                        |20, 17                          |                                |
|                        |DeformConv                                              |                        |22, 19                          |                                |
|                        |DepthToSpace                                            |                        |13, 11, 1                       |                                |
|                        |DequantizeLinear                                        |                        |21, 19, 13, 10                  |                                |
|                        |Det                                                     |                        |22, 11                          |                                |
|                        |Div                                                     |                        |14, 13, 7, 6, 1                 |                                |
|                        |Dropout                                                 |                        |22, 13, 12, 10, 7, 6, 1         |                                |
|                        |Einsum                                                  |                        |12                              |                                |
|                        |Equal                                                   |                        |19, 13, 11, 7, 1                |                                |
|                        |Erf                                                     |                        |13, 9                           |                                |
|                        |Exp                                                     |                        |13, 6, 1                        |                                |
|                        |Expand                                                  |                        |13, 8                           |                                |
|                        |EyeLike                                                 |                        |22, 9                           |                                |
|                        |Flatten                                                 |                        |21, 13, 11, 9, 1                |                                |
|                        |Floor                                                   |                        |13, 6, 1                        |                                |
|                        |GRU                                                     |                        |22, 14, 7, 3, 1                 |                                |
|                        |Gather                                                  |                        |13, 11, 1                       |                                |
|                        |GatherElements                                          |                        |13, 11                          |                                |
|                        |GatherND                                                |                        |13, 12, 11                      |                                |
|                        |Gemm                                                    |                        |13, 11, 9, 7, 6, 1              |                                |
|                        |GlobalAveragePool                                       |                        |22, 1                           |                                |
|                        |GlobalLpPool                                            |                        |22, 2, 1                        |                                |
|                        |GlobalMaxPool                                           |                        |22, 1                           |                                |
|                        |Greater                                                 |                        |13, 9, 7, 1                     |                                |
|                        |GridSample                                              |                        |22, 20, 16                      |                                |
|                        |Hardmax                                                 |                        |13, 11, 1                       |                                |
|                        |Identity                                                |                        |21, 19, 16, 14, 13, 1           |                                |
|                        |If                                                      |                        |21, 19, 16, 13, 11, 1           |                                |
|                        |ImageDecoder                                            |                        |20                              |                                |
|                        |ImageScaler                                             |                        |                                |                                |
|                        |InstanceNormalization                                   |                        |22, 6, 1                        |                                |
|                        |IsFinite                                                |                        |                                |                                |
|                        |IsInf                                                   |                        |20, 10                          |                                |
|                        |IsNaN                                                   |                        |20, 13, 9                       |                                |
|                        |LRN                                                     |                        |13, 1                           |                                |
|                        |LSTM                                                    |                        |22, 14, 7, 1                    |                                |
|                        |Less                                                    |                        |13, 9, 7, 1                     |                                |
|                        |Log                                                     |                        |13, 6, 1                        |                                |
|                        |Loop                                                    |                        |21, 19, 16, 13, 11, 1           |                                |
|                        |LpNormalization                                         |                        |22, 1                           |                                |
|                        |LpPool                                                  |                        |22, 18, 11, 2, 1                |                                |
|                        |MatMul                                                  |                        |13, 9, 1                        |                                |
|                        |MatMulInteger                                           |                        |10                              |                                |
|                        |Max                                                     |                        |13, 12, 8, 6, 1                 |                                |
|                        |MaxPool                                                 |                        |22, 12, 11, 10, 8, 1            |                                |
|                        |MaxRoiPool                                              |                        |22, 1                           |                                |
|                        |MaxUnpool                                               |                        |22, 11, 9                       |                                |
|                        |Mean                                                    |                        |13, 8, 6, 1                     |                                |
|                        |MelWeightMatrix                                         |                        |17                              |                                |
|                        |Min                                                     |                        |13, 12, 8, 6, 1                 |                                |
|                        |Mod                                                     |                        |13, 10                          |                                |
|                        |Mul                                                     |                        |14, 13, 7, 6, 1                 |                                |
|                        |Multinomial                                             |                        |22, 7                           |                                |
|                        |Neg                                                     |                        |13, 6, 1                        |                                |
|                        |NonMaxSuppression                                       |                        |11, 10                          |                                |
|                        |NonZero                                                 |                        |13, 9                           |                                |
|                        |Not                                                     |                        |1                               |                                |
|                        |OneHot                                                  |                        |11, 9                           |                                |
|                        |Optional                                                |                        |15                              |                                |
|                        |OptionalGetElement                                      |                        |18, 15                          |                                |
|                        |OptionalHasElement                                      |                        |18, 15                          |                                |
|                        |Or                                                      |                        |7, 1                            |                                |
|                        |Pad                                                     |                        |21, 19, 18, 13, 11, 2, 1        |                                |
|                        |Pow                                                     |                        |15, 13, 12, 7, 1                |                                |
|                        |QLinearConv                                             |                        |10                              |                                |
|                        |QLinearMatMul                                           |                        |21, 10                          |                                |
|                        |QuantizeLinear                                          |                        |21, 19, 13, 10                  |                                |
|                        |RNN                                                     |                        |22, 14, 7, 1                    |                                |
|                        |RandomNormal                                            |                        |22, 1                           |                                |
|                        |RandomNormalLike                                        |                        |22, 1                           |                                |
|                        |RandomUniform                                           |                        |22, 1                           |                                |
|                        |RandomUniformLike                                       |                        |22, 1                           |                                |
|                        |Reciprocal                                              |                        |13, 6, 1                        |                                |
|                        |ReduceMax                                               |                        |20, 18, 13, 12, 11, 1           |                                |
|                        |ReduceMean                                              |                        |18, 13, 11, 1                   |                                |
|                        |ReduceMin                                               |                        |20, 18, 13, 12, 11, 1           |                                |
|                        |ReduceProd                                              |                        |18, 13, 11, 1                   |                                |
|                        |ReduceSum                                               |                        |13, 11, 1                       |                                |
|                        |RegexFullMatch                                          |                        |20                              |                                |
|                        |Reshape                                                 |                        |21, 19, 14, 13, 5, 1            |                                |
|                        |Resize                                                  |                        |19, 18, 13, 11, 10              |                                |
|                        |ReverseSequence                                         |                        |10                              |                                |
|                        |RoiAlign                                                |                        |22, 16, 10                      |                                |
|                        |Round                                                   |                        |22, 11                          |                                |
|                        |STFT                                                    |                        |17                              |                                |
|                        |Scan                                                    |                        |21, 19, 16, 11, 9, 8            |                                |
|                        |Scatter                                                 |                        |11, 9                           |                                |
|                        |ScatterElements                                         |                        |18, 16, 13, 11                  |                                |
|                        |ScatterND                                               |                        |18, 16, 13, 11                  |                                |
|                        |SequenceAt                                              |                        |11                              |                                |
|                        |SequenceConstruct                                       |                        |11                              |                                |
|                        |SequenceEmpty                                           |                        |11                              |                                |
|                        |SequenceErase                                           |                        |11                              |                                |
|                        |SequenceInsert                                          |                        |11                              |                                |
|                        |SequenceLength                                          |                        |11                              |                                |
|                        |Shape                                                   |                        |21, 19, 15, 13, 1               |                                |
|                        |Sigmoid                                                 |                        |13, 6, 1                        |                                |
|                        |Sign                                                    |                        |13, 9                           |                                |
|                        |Sin                                                     |                        |22, 7                           |                                |
|                        |Sinh                                                    |                        |22, 9                           |                                |
|                        |Size                                                    |                        |21, 19, 13, 1                   |                                |
|                        |Slice                                                   |                        |13, 11, 10, 1                   |                                |
|                        |SpaceToDepth                                            |                        |13, 1                           |                                |
|                        |Split                                                   |                        |18, 13, 11, 2, 1                |                                |
|                        |SplitToSequence                                         |                        |11                              |                                |
|                        |Sqrt                                                    |                        |13, 6, 1                        |                                |
|                        |Squeeze                                                 |                        |21, 13, 11, 1                   |                                |
|                        |StringConcat                                            |                        |20                              |                                |
|                        |StringNormalizer                                        |                        |10                              |                                |
|                        |StringSplit                                             |                        |20                              |                                |
|                        |Sub                                                     |                        |14, 13, 7, 6, 1                 |                                |
|                        |Sum                                                     |                        |13, 8, 6, 1                     |                                |
|                        |Tan                                                     |                        |22, 7                           |                                |
|                        |Tanh                                                    |                        |13, 6, 1                        |                                |
|                        |TfIdfVectorizer                                         |                        |9                               |                                |
|                        |Tile                                                    |                        |13, 6, 1                        |                                |
|                        |TopK                                                    |                        |11, 10, 1                       |                                |
|                        |Transpose                                               |                        |21, 13, 1                       |                                |
|                        |Trilu                                                   |                        |14                              |                                |
|                        |Unique                                                  |                        |11                              |                                |
|                        |Unsqueeze                                               |                        |21, 13, 11, 1                   |                                |
|                        |Upsample                                                |                        |10, 9, 7                        |                                |
|                        |Where                                                   |                        |16, 9                           |                                |
|                        |Xor                                                     |                        |7, 1                            |                                |
|                        |AffineGrid                                              |                        |20                              |                                |
|                        |Bernoulli                                               |                        |22, 15                          |                                |
|                        |BlackmanWindow                                          |                        |17                              |                                |
|                        |CastLike                                                |                        |21, 19, 15                      |                                |
|                        |Celu                                                    |                        |12                              |                                |
|                        |CenterCropPad                                           |                        |18                              |                                |
|                        |Clip                                                    |                        |13, 12, 11, 6, 1                |                                |
|                        |DynamicQuantizeLinear                                   |                        |11                              |                                |
|                        |Elu                                                     |                        |22, 6, 1                        |                                |
|                        |Gelu                                                    |                        |20                              |                                |
|                        |GreaterOrEqual                                          |                        |16, 12                          |                                |
|                        |GroupNormalization                                      |                        |21, 18                          |                                |
|                        |HammingWindow                                           |                        |17                              |                                |
|                        |HannWindow                                              |                        |17                              |                                |
|                        |HardSigmoid                                             |                        |22, 6, 1                        |                                |
|                        |HardSwish                                               |                        |22, 14                          |                                |
|                        |LayerNormalization                                      |                        |17                              |                                |
|                        |LeakyRelu                                               |                        |16, 6, 1                        |                                |
|                        |LessOrEqual                                             |                        |16, 12                          |                                |
|                        |LogSoftmax                                              |                        |13, 11, 1                       |                                |
|                        |MeanVarianceNormalization                               |                        |13, 9                           |                                |
|                        |Mish                                                    |                        |22, 18                          |                                |
|                        |NegativeLogLikelihoodLoss                               |                        |22, 13, 12                      |                                |
|                        |PRelu                                                   |                        |16, 9, 7, 6, 1                  |                                |
|                        |Range                                                   |                        |11                              |                                |
|                        |ReduceL1                                                |                        |18, 13, 11, 1                   |                                |
|                        |ReduceL2                                                |                        |18, 13, 11, 1                   |                                |
|                        |ReduceLogSum                                            |                        |18, 13, 11, 1                   |                                |
|                        |ReduceLogSumExp                                         |                        |18, 13, 11, 1                   |                                |
|                        |ReduceSumSquare                                         |                        |18, 13, 11, 1                   |                                |
|                        |Relu                                                    |                        |14, 13, 6, 1                    |                                |
|                        |Selu                                                    |                        |22, 6, 1                        |                                |
|                        |SequenceMap                                             |                        |17                              |                                |
|                        |Shrink                                                  |                        |9                               |                                |
|                        |SimplifiedLayerNormalization                            |                        |1                               |                                |
|                        |Softmax                                                 |                        |13, 11, 1                       |                                |
|                        |SoftmaxCrossEntropyLoss                                 |                        |13, 12                          |                                |
|                        |Softplus                                                |                        |22, 1                           |                                |
|                        |Softsign                                                |                        |22, 1                           |                                |
|                        |ThresholdedRelu                                         |                        |22, 10                          |                                |
|com.microsoft           |Attention                                               |                        |1                               |                                |
|com.microsoft           |AttnLSTM                                                |                        |1                               |                                |
|com.microsoft           |BeamSearch                                              |                        |1                               |                                |
|com.microsoft           |BiasAdd                                                 |                        |1                               |                                |
|com.microsoft           |BiasDropout                                             |                        |1                               |                                |
|com.microsoft           |BiasGelu                                                |                        |1                               |                                |
|com.microsoft           |BiasSoftmax                                             |                        |1                               |                                |
|com.microsoft           |BiasSplitGelu                                           |                        |1                               |                                |
|com.microsoft           |BifurcationDetector                                     |                        |1                               |                                |
|com.microsoft           |BitmaskBiasDropout                                      |                        |1                               |                                |
|com.microsoft           |BitmaskDropout                                          |                        |1                               |                                |
|com.microsoft           |CDist                                                   |                        |1                               |                                |
|com.microsoft           |ComplexMul                                              |                        |1                               |                                |
|com.microsoft           |ComplexMulConj                                          |                        |1                               |                                |
|com.microsoft           |ConvTransposeWithDynamicPads                            |                        |1                               |                                |
|com.microsoft           |CropAndResize                                           |                        |1                               |                                |
|com.microsoft           |DecoderAttention                                        |                        |1                               |                                |
|com.microsoft           |DecoderMaskedMultiHeadAttention                         |                        |1                               |                                |
|com.microsoft           |DecoderMaskedSelfAttention                              |                        |1                               |                                |
|com.microsoft           |DequantizeBFP                                           |                        |1                               |                                |
|com.microsoft           |DequantizeLinear                                        |                        |1                               |                                |
|com.microsoft           |DequantizeWithOrder                                     |                        |1                               |                                |
|com.microsoft           |DynamicQuantizeLSTM                                     |                        |1                               |                                |
|com.microsoft           |DynamicQuantizeMatMul                                   |                        |1                               |                                |
|com.microsoft           |DynamicTimeWarping                                      |                        |1                               |                                |
|com.microsoft           |EPContext                                               |                        |1                               |                                |
|com.microsoft           |EmbedLayerNormalization                                 |                        |1                               |                                |
|com.microsoft           |ExpandDims                                              |                        |1                               |                                |
|com.microsoft           |FastGelu                                                |                        |1                               |                                |
|com.microsoft           |FusedConv                                               |                        |1                               |                                |
|com.microsoft           |FusedGemm                                               |                        |1                               |                                |
|com.microsoft           |FusedMatMul                                             |                        |1                               |                                |
|com.microsoft           |FusedMatMulActivation                                   |                        |1                               |                                |
|com.microsoft           |GatedRelativePositionBias                               |                        |1                               |                                |
|com.microsoft           |GatherND                                                |                        |1                               |                                |
|com.microsoft           |Gelu                                                    |                        |1                               |                                |
|com.microsoft           |GemmFastGelu                                            |                        |1                               |                                |
|com.microsoft           |GemmFloat8                                              |                        |1                               |                                |
|com.microsoft           |GemmaRotaryEmbedding                                    |                        |1                               |                                |
|com.microsoft           |GreedySearch                                            |                        |1                               |                                |
|com.microsoft           |GridSample                                              |                        |1                               |                                |
|com.microsoft           |GroupNorm                                               |                        |1                               |                                |
|com.microsoft           |GroupQueryAttention                                     |                        |1                               |                                |
|com.microsoft           |Inverse                                                 |                        |1                               |                                |
|com.microsoft           |Irfft                                                   |                        |1                               |                                |
|com.microsoft           |LongformerAttention                                     |                        |1                               |                                |
|com.microsoft           |MatMulBnb4                                              |                        |1                               |                                |
|com.microsoft           |MatMulFpQ4                                              |                        |1                               |                                |
|com.microsoft           |MatMulInteger16                                         |                        |1                               |                                |
|com.microsoft           |MatMulIntegerToFloat                                    |                        |1                               |                                |
|com.microsoft           |MatMulNBits                                             |                        |1                               |                                |
|com.microsoft           |MaxpoolWithMask                                         |                        |1                               |                                |
|com.microsoft           |MoE                                                     |                        |1                               |                                |
|com.microsoft           |MulInteger                                              |                        |1                               |                                |
|com.microsoft           |MultiHeadAttention                                      |                        |1                               |                                |
|com.microsoft           |MurmurHash3                                             |                        |1                               |                                |
|com.microsoft           |NGramRepeatBlock                                        |                        |1                               |                                |
|com.microsoft           |NhwcConv                                                |                        |1                               |                                |
|com.microsoft           |NhwcFusedConv                                           |                        |1                               |                                |
|com.microsoft           |NhwcMaxPool                                             |                        |1                               |                                |
|com.microsoft           |PackedAttention                                         |                        |1                               |                                |
|com.microsoft           |PackedMultiHeadAttention                                |                        |1                               |                                |
|com.microsoft           |Pad                                                     |                        |1                               |                                |
|com.microsoft           |QAttention                                              |                        |1                               |                                |
|com.microsoft           |QGemm                                                   |                        |1                               |                                |
|com.microsoft           |QLinearAdd                                              |                        |1                               |                                |
|com.microsoft           |QLinearAveragePool                                      |                        |1                               |                                |
|com.microsoft           |QLinearConcat                                           |                        |1                               |                                |
|com.microsoft           |QLinearConv                                             |                        |1                               |                                |
|com.microsoft           |QLinearGlobalAveragePool                                |                        |1                               |                                |
|com.microsoft           |QLinearLeakyRelu                                        |                        |1                               |                                |
|com.microsoft           |QLinearMul                                              |                        |1                               |                                |
|com.microsoft           |QLinearReduceMean                                       |                        |1                               |                                |
|com.microsoft           |QLinearSigmoid                                          |                        |1                               |                                |
|com.microsoft           |QLinearSoftmax                                          |                        |1                               |                                |
|com.microsoft           |QLinearWhere                                            |                        |1                               |                                |
|com.microsoft           |QMoE                                                    |                        |1                               |                                |
|com.microsoft           |QOrderedAttention                                       |                        |1                               |                                |
|com.microsoft           |QOrderedGelu                                            |                        |1                               |                                |
|com.microsoft           |QOrderedLayerNormalization                              |                        |1                               |                                |
|com.microsoft           |QOrderedLongformerAttention                             |                        |1                               |                                |
|com.microsoft           |QOrderedMatMul                                          |                        |1                               |                                |
|com.microsoft           |QuantizeBFP                                             |                        |1                               |                                |
|com.microsoft           |QuantizeLinear                                          |                        |1                               |                                |
|com.microsoft           |QuantizeWithOrder                                       |                        |1                               |                                |
|com.microsoft           |QuickGelu                                               |                        |1                               |                                |
|com.microsoft           |Range                                                   |                        |1                               |                                |
|com.microsoft           |ReduceSumInteger                                        |                        |1                               |                                |
|com.microsoft           |RelativePositionBias                                    |                        |1                               |                                |
|com.microsoft           |RemovePadding                                           |                        |1                               |                                |
|com.microsoft           |RestorePadding                                          |                        |1                               |                                |
|com.microsoft           |Rfft                                                    |                        |1                               |                                |
|com.microsoft           |RotaryEmbedding                                         |                        |1                               |                                |
|com.microsoft           |SampleOp                                                |                        |1                               |                                |
|com.microsoft           |Sampling                                                |                        |1                               |                                |
|com.microsoft           |SimplifiedLayerNormalization                            |                        |1                               |                                |
|com.microsoft           |SkipGroupNorm                                           |                        |1                               |                                |
|com.microsoft           |SkipLayerNormalization                                  |                        |1                               |                                |
|com.microsoft           |SkipSimplifiedLayerNormalization                        |                        |1                               |                                |
|com.microsoft           |Snpe                                                    |                        |1                               |                                |
|com.microsoft           |SparseAttention                                         |                        |1                               |                                |
|com.microsoft           |SparseToDenseMatMul                                     |                        |1                               |                                |
|com.microsoft           |Tokenizer                                               |                        |1                               |                                |
|com.microsoft           |TorchEmbedding                                          |                        |1                               |                                |
|com.microsoft           |TransposeMatMul                                         |                        |1                               |                                |
|com.microsoft           |Trilu                                                   |                        |1                               |                                |
|com.microsoft           |UnfoldTensor                                            |                        |1                               |                                |
|com.microsoft           |Unique                                                  |                        |1                               |                                |
|com.microsoft           |WhisperBeamSearch                                       |                        |1                               |                                |
|com.microsoft           |WordConvEmbedding                                       |                        |1                               |                                |
|org.openvinotoolkit     |DeformableConv2D                                        |1                       |1                               |                                |
|org.openvinotoolkit     |DetectionOutput                                         |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronDetectionOutput                    |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronGenerateProposalsSingleImage       |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronGroupNorm                          |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronPriorGridGenerator                 |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronROIFeatureExtractor                |1                       |1                               |                                |
|org.openvinotoolkit     |ExperimentalDetectronTopKROIs                           |1                       |1                               |                                |
|org.openvinotoolkit     |FakeQuantize                                            |1                       |1                               |                                |
|org.openvinotoolkit     |GenerateProposals                                       |1                       |1                               |                                |
|org.openvinotoolkit     |GroupNorm                                               |1                       |1                               |                                |
|org.openvinotoolkit     |Normalize                                               |1                       |1                               |                                |
|org.openvinotoolkit     |PriorBox                                                |1                       |1                               |                                |
|org.openvinotoolkit     |PriorBoxClustered                                       |1                       |1                               |                                |
|org.openvinotoolkit     |Swish                                                   |1                       |1                               |                                |
|org.pytorch.aten        |adaptive_avg_pool2d                                     |1                       |1                               |                                |
|mmdeploy                |MMCVRoIAlignRotated                                     |1                       |1                               |                                |
|mmdeploy                |NMSRotated                                              |1                       |1                               |                                |
