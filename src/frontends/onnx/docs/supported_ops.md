# ONNX Operations Supported by OpenVINO ONNX Frontend

Here is a table of operations available by the ONNX Frontend.

OpenVINO provides support for operations of Default Opset (empty in table below), other operations are supported by community.

| Domain                 | Operation Name                                         | Supported Version      | Defined Version                | Limitation                     |
|------------------------|--------------------------------------------------------|------------------------|--------------------------------|--------------------------------|
|                        |ATen                                                    |1                       |                                |                                |
|                        |Abs                                                     |13, 6, 1                |13, 6, 1                        |                                |
|                        |Acos                                                    |1                       |22, 7                           |                                |
|                        |Acosh                                                   |1                       |22, 9                           |                                |
|                        |Add                                                     |14, 13, 7, 6, 1         |14, 13, 7, 6, 1                 |                                |
|                        |Affine                                                  |1                       |                                |                                |
|                        |AffineGrid                                              |                        |20                              |                                |
|                        |And                                                     |7, 1                    |7, 1                            |                                |
|                        |ArgMax                                                  |12, 1                   |13, 12, 11, 1                   |                                |
|                        |ArgMin                                                  |12, 1                   |13, 12, 11, 1                   |                                |
|                        |Asin                                                    |1                       |22, 7                           |                                |
|                        |Asinh                                                   |1                       |22, 9                           |                                |
|                        |Atan                                                    |1                       |22, 7                           |                                |
|                        |Atanh                                                   |1                       |22, 9                           |                                |
|                        |AveragePool                                             |1                       |22, 19, 11, 10, 7, 1            |                                |
|                        |BatchNormalization                                      |14, 7, 1                |15, 14, 9, 7, 6, 1              |                                |
|                        |Bernoulli                                               |                        |22, 15                          |                                |
|                        |BitShift                                                |1                       |11                              |                                |
|                        |BitwiseAnd                                              |1                       |18                              |                                |
|                        |BitwiseNot                                              |1                       |18                              |                                |
|                        |BitwiseOr                                               |1                       |18                              |                                |
|                        |BitwiseXor                                              |1                       |18                              |                                |
|                        |BlackmanWindow                                          |1                       |17                              |                                |
|                        |Cast                                                    |1                       |21, 19, 13, 9, 6, 1             |                                |
|                        |CastLike                                                |1                       |21, 19, 15                      |                                |
|                        |Ceil                                                    |1                       |13, 6, 1                        |                                |
|                        |Celu                                                    |1                       |12                              |                                |
|                        |CenterCropPad                                           |                        |18                              |                                |
|                        |Clip                                                    |11, 1                   |13, 12, 11, 6, 1                |                                |
|                        |Col2Im                                                  |                        |18                              |                                |
|                        |Compress                                                |1                       |11, 9                           |                                |
|                        |Concat                                                  |1                       |13, 11, 4, 1                    |                                |
|                        |ConcatFromSequence                                      |                        |11                              |                                |
|                        |Constant                                                |13, 1                   |21, 19, 13, 12, 11, 9, 1        |                                |
|                        |ConstantFill                                            |1                       |                                |                                |
|                        |ConstantOfShape                                         |1                       |21, 20, 9                       |                                |
|                        |Conv                                                    |1                       |22, 11, 1                       |                                |
|                        |ConvInteger                                             |1                       |10                              |                                |
|                        |ConvTranspose                                           |1                       |22, 11, 1                       |                                |
|                        |Cos                                                     |1                       |22, 7                           |                                |
|                        |Cosh                                                    |1                       |22, 9                           |                                |
|                        |Crop                                                    |1                       |                                |                                |
|                        |CumSum                                                  |1                       |14, 11                          |                                |
|                        |DFT                                                     |1                       |20, 17                          |                                |
|                        |DeformConv                                              |                        |22, 19                          |                                |
|                        |DepthToSpace                                            |1                       |13, 11, 1                       |                                |
|                        |DequantizeLinear                                        |21, 19, 13, 1           |21, 19, 13, 10                  |                                |
|                        |Det                                                     |                        |22, 11                          |                                |
|                        |Div                                                     |7, 1                    |14, 13, 7, 6, 1                 |                                |
|                        |Dropout                                                 |12, 7, 1                |22, 13, 12, 10, 7, 6, 1         |                                |
|                        |DynamicQuantizeLinear                                   |1                       |11                              |                                |
|                        |Einsum                                                  |1                       |12                              |                                |
|                        |Elu                                                     |1                       |22, 6, 1                        |                                |
|                        |Equal                                                   |1                       |19, 13, 11, 7, 1                |                                |
|                        |Erf                                                     |1                       |13, 9                           |                                |
|                        |Exp                                                     |1                       |13, 6, 1                        |                                |
|                        |Expand                                                  |1                       |13, 8                           |                                |
|                        |EyeLike                                                 |1                       |22, 9                           |                                |
|                        |Flatten                                                 |1                       |21, 13, 11, 9, 1                |                                |
|                        |Floor                                                   |1                       |13, 6, 1                        |                                |
|                        |GRU                                                     |1                       |22, 14, 7, 3, 1                 |                                |
|                        |Gather                                                  |1                       |13, 11, 1                       |                                |
|                        |GatherElements                                          |1                       |13, 11                          |                                |
|                        |GatherND                                                |1                       |13, 12, 11                      |                                |
|                        |Gelu                                                    |1                       |20                              |                                |
|                        |Gemm                                                    |6, 1                    |13, 11, 9, 7, 6, 1              |                                |
|                        |GlobalAveragePool                                       |1                       |22, 1                           |                                |
|                        |GlobalLpPool                                            |1                       |22, 2, 1                        |                                |
|                        |GlobalMaxPool                                           |1                       |22, 1                           |                                |
|                        |Greater                                                 |1                       |13, 9, 7, 1                     |                                |
|                        |GreaterOrEqual                                          |16, 1                   |16, 12                          |                                |
|                        |GridSample                                              |1                       |22, 20, 16                      |                                |
|                        |GroupNormalization                                      |1                       |21, 18                          |                                |
|                        |HammingWindow                                           |1                       |17                              |                                |
|                        |HannWindow                                              |1                       |17                              |                                |
|                        |HardSigmoid                                             |1                       |22, 6, 1                        |                                |
|                        |HardSwish                                               |1                       |22, 14                          |                                |
|                        |Hardmax                                                 |13, 1                   |13, 11, 1                       |                                |
|                        |Identity                                                |1                       |21, 19, 16, 14, 13, 1           |                                |
|                        |If                                                      |1                       |21, 19, 16, 13, 11, 1           |                                |
|                        |ImageDecoder                                            |                        |20                              |                                |
|                        |ImageScaler                                             |1                       |                                |                                |
|                        |InstanceNormalization                                   |1                       |22, 6, 1                        |                                |
|                        |IsFinite                                                |1                       |                                |                                |
|                        |IsInf                                                   |1                       |20, 10                          |                                |
|                        |IsNaN                                                   |1                       |20, 13, 9                       |                                |
|                        |LRN                                                     |1                       |13, 1                           |                                |
|                        |LSTM                                                    |1                       |22, 14, 7, 1                    |                                |
|                        |LayerNormalization                                      |1                       |17                              |                                |
|                        |LeakyRelu                                               |1                       |16, 6, 1                        |                                |
|                        |Less                                                    |1                       |13, 9, 7, 1                     |                                |
|                        |LessOrEqual                                             |16, 1                   |16, 12                          |                                |
|                        |Log                                                     |1                       |13, 6, 1                        |                                |
|                        |LogSoftmax                                              |13, 1                   |13, 11, 1                       |                                |
|                        |Loop                                                    |1                       |21, 19, 16, 13, 11, 1           |                                |
|                        |LpNormalization                                         |1                       |22, 1                           |                                |
|                        |LpPool                                                  |                        |22, 18, 11, 2, 1                |                                |
|                        |MatMul                                                  |1                       |13, 9, 1                        |                                |
|                        |MatMulInteger                                           |1                       |10                              |                                |
|                        |Max                                                     |8, 1                    |13, 12, 8, 6, 1                 |                                |
|                        |MaxPool                                                 |8, 1                    |22, 12, 11, 10, 8, 1            |                                |
|                        |MaxRoiPool                                              |1                       |22, 1                           |                                |
|                        |MaxUnpool                                               |                        |22, 11, 9                       |                                |
|                        |Mean                                                    |1                       |13, 8, 6, 1                     |                                |
|                        |MeanVarianceNormalization                               |9, 1                    |13, 9                           |                                |
|                        |MelWeightMatrix                                         |                        |17                              |                                |
|                        |Min                                                     |8, 1                    |13, 12, 8, 6, 1                 |                                |
|                        |Mish                                                    |1                       |22, 18                          |                                |
|                        |Mod                                                     |1                       |13, 10                          |                                |
|                        |Mul                                                     |7, 1                    |14, 13, 7, 6, 1                 |                                |
|                        |Multinomial                                             |1                       |22, 7                           |                                |
|                        |Neg                                                     |1                       |13, 6, 1                        |                                |
|                        |NegativeLogLikelihoodLoss                               |                        |22, 13, 12                      |                                |
|                        |NonMaxSuppression                                       |1                       |11, 10                          |                                |
|                        |NonZero                                                 |1                       |13, 9                           |                                |
|                        |Not                                                     |1                       |1                               |                                |
|                        |OneHot                                                  |1                       |11, 9                           |                                |
|                        |Optional                                                |                        |15                              |                                |
|                        |OptionalGetElement                                      |                        |18, 15                          |                                |
|                        |OptionalHasElement                                      |                        |18, 15                          |                                |
|                        |Or                                                      |1                       |7, 1                            |                                |
|                        |PRelu                                                   |1                       |16, 9, 7, 6, 1                  |                                |
|                        |Pad                                                     |11, 1                   |21, 19, 18, 13, 11, 2, 1        |                                |
|                        |Pow                                                     |1                       |15, 13, 12, 7, 1                |                                |
|                        |QLinearConv                                             |1                       |10                              |                                |
|                        |QLinearMatMul                                           |1                       |21, 10                          |                                |
|                        |QuantizeLinear                                          |13, 1                   |21, 19, 13, 10                  |                                |
|                        |RNN                                                     |1                       |22, 14, 7, 1                    |                                |
|                        |RandomNormal                                            |1                       |22, 1                           |                                |
|                        |RandomNormalLike                                        |1                       |22, 1                           |                                |
|                        |RandomUniform                                           |1                       |22, 1                           |                                |
|                        |RandomUniformLike                                       |1                       |22, 1                           |                                |
|                        |Range                                                   |1                       |11                              |                                |
|                        |Reciprocal                                              |1                       |13, 6, 1                        |                                |
|                        |ReduceL1                                                |18, 1                   |18, 13, 11, 1                   |                                |
|                        |ReduceL2                                                |18, 13, 1               |18, 13, 11, 1                   |                                |
|                        |ReduceLogSum                                            |18, 1                   |18, 13, 11, 1                   |                                |
|                        |ReduceLogSumExp                                         |18, 13, 1               |18, 13, 11, 1                   |                                |
|                        |ReduceMax                                               |20, 18, 13, 1           |20, 18, 13, 12, 11, 1           |                                |
|                        |ReduceMean                                              |18, 13, 1               |18, 13, 11, 1                   |                                |
|                        |ReduceMin                                               |20, 18, 13, 1           |20, 18, 13, 12, 11, 1           |                                |
|                        |ReduceProd                                              |18, 13, 1               |18, 13, 11, 1                   |                                |
|                        |ReduceSum                                               |13, 1                   |13, 11, 1                       |                                |
|                        |ReduceSumSquare                                         |18, 13, 1               |18, 13, 11, 1                   |                                |
|                        |RegexFullMatch                                          |                        |20                              |                                |
|                        |Relu                                                    |1                       |14, 13, 6, 1                    |                                |
|                        |Reshape                                                 |1                       |21, 19, 14, 13, 5, 1            |                                |
|                        |Resize                                                  |11, 1                   |19, 18, 13, 11, 10              |                                |
|                        |ReverseSequence                                         |1                       |10                              |                                |
|                        |RoiAlign                                                |16, 1                   |22, 16, 10                      |                                |
|                        |Round                                                   |1                       |22, 11                          |                                |
|                        |STFT                                                    |1                       |17                              |                                |
|                        |Scan                                                    |9, 1                    |21, 19, 16, 11, 9, 8            |                                |
|                        |Scatter                                                 |1                       |11, 9                           |                                |
|                        |ScatterElements                                         |1                       |18, 16, 13, 11                  |                                |
|                        |ScatterND                                               |1                       |18, 16, 13, 11                  |                                |
|                        |Selu                                                    |1                       |22, 6, 1                        |                                |
|                        |SequenceAt                                              |                        |11                              |                                |
|                        |SequenceConstruct                                       |                        |11                              |                                |
|                        |SequenceEmpty                                           |                        |11                              |                                |
|                        |SequenceErase                                           |                        |11                              |                                |
|                        |SequenceInsert                                          |                        |11                              |                                |
|                        |SequenceLength                                          |                        |11                              |                                |
|                        |SequenceMap                                             |                        |17                              |                                |
|                        |Shape                                                   |15, 1                   |21, 19, 15, 13, 1               |                                |
|                        |Shrink                                                  |1                       |9                               |                                |
|                        |Sigmoid                                                 |1                       |13, 6, 1                        |                                |
|                        |Sign                                                    |1                       |13, 9                           |                                |
|                        |SimplifiedLayerNormalization                            |                        |1                               |                                |
|                        |Sin                                                     |1                       |22, 7                           |                                |
|                        |Sinh                                                    |1                       |22, 9                           |                                |
|                        |Size                                                    |1                       |21, 19, 13, 1                   |                                |
|                        |Slice                                                   |10, 1                   |13, 11, 10, 1                   |                                |
|                        |Softmax                                                 |13, 11, 1               |13, 11, 1                       |                                |
|                        |SoftmaxCrossEntropyLoss                                 |                        |13, 12                          |                                |
|                        |Softplus                                                |1                       |22, 1                           |                                |
|                        |Softsign                                                |1                       |22, 1                           |                                |
|                        |SpaceToDepth                                            |1                       |13, 1                           |                                |
|                        |Split                                                   |13, 1                   |18, 13, 11, 2, 1                |                                |
|                        |SplitToSequence                                         |                        |11                              |                                |
|                        |Sqrt                                                    |1                       |13, 6, 1                        |                                |
|                        |Squeeze                                                 |13, 1                   |21, 13, 11, 1                   |                                |
|                        |StringConcat                                            |                        |20                              |                                |
|                        |StringNormalizer                                        |                        |10                              |                                |
|                        |StringSplit                                             |                        |20                              |                                |
|                        |Sub                                                     |7, 1                    |14, 13, 7, 6, 1                 |                                |
|                        |Sum                                                     |8, 1                    |13, 8, 6, 1                     |                                |
|                        |Tan                                                     |1                       |22, 7                           |                                |
|                        |Tanh                                                    |1                       |13, 6, 1                        |                                |
|                        |TfIdfVectorizer                                         |                        |9                               |                                |
|                        |ThresholdedRelu                                         |1                       |22, 10                          |                                |
|                        |Tile                                                    |1                       |13, 6, 1                        |                                |
|                        |TopK                                                    |11, 10, 1               |11, 10, 1                       |                                |
|                        |Transpose                                               |1                       |21, 13, 1                       |                                |
|                        |Trilu                                                   |1                       |14                              |                                |
|                        |Unique                                                  |1                       |11                              |                                |
|                        |Unsqueeze                                               |13, 1                   |21, 13, 11, 1                   |                                |
|                        |Upsample                                                |9, 7, 1                 |10, 9, 7                        |                                |
|                        |Where                                                   |1                       |16, 9                           |                                |
|                        |Xor                                                     |1                       |7, 1                            |                                |
|com.microsoft           |Attention                                               |1                       |1                               |                                |
|com.microsoft           |AttnLSTM                                                |                        |1                               |                                |
|com.microsoft           |BeamSearch                                              |                        |1                               |                                |
|com.microsoft           |BiasAdd                                                 |1                       |1                               |                                |
|com.microsoft           |BiasDropout                                             |                        |1                               |                                |
|com.microsoft           |BiasGelu                                                |1                       |1                               |                                |
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
|com.microsoft           |DequantizeLinear                                        |1                       |1                               |                                |
|com.microsoft           |DequantizeWithOrder                                     |                        |1                               |                                |
|com.microsoft           |DynamicQuantizeLSTM                                     |                        |1                               |                                |
|com.microsoft           |DynamicQuantizeMatMul                                   |1                       |1                               |                                |
|com.microsoft           |DynamicTimeWarping                                      |                        |1                               |                                |
|com.microsoft           |EPContext                                               |                        |1                               |                                |
|com.microsoft           |EmbedLayerNormalization                                 |1                       |1                               |                                |
|com.microsoft           |ExpandDims                                              |                        |1                               |                                |
|com.microsoft           |FastGelu                                                |1                       |1                               |                                |
|com.microsoft           |FusedConv                                               |1                       |1                               |                                |
|com.microsoft           |FusedGemm                                               |1                       |1                               |                                |
|com.microsoft           |FusedMatMul                                             |1                       |1                               |                                |
|com.microsoft           |FusedMatMulActivation                                   |                        |1                               |                                |
|com.microsoft           |GatedRelativePositionBias                               |                        |1                               |                                |
|com.microsoft           |GatherND                                                |1                       |1                               |                                |
|com.microsoft           |Gelu                                                    |1                       |1                               |                                |
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
|com.microsoft           |MatMulIntegerToFloat                                    |1                       |1                               |                                |
|com.microsoft           |MatMulNBits                                             |1                       |1                               |                                |
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
|com.microsoft           |Pad                                                     |1                       |1                               |                                |
|com.microsoft           |QAttention                                              |                        |1                               |                                |
|com.microsoft           |QGemm                                                   |                        |1                               |                                |
|com.microsoft           |QLinearAdd                                              |1                       |1                               |                                |
|com.microsoft           |QLinearAveragePool                                      |1                       |1                               |                                |
|com.microsoft           |QLinearConcat                                           |                        |1                               |                                |
|com.microsoft           |QLinearConv                                             |                        |1                               |                                |
|com.microsoft           |QLinearGlobalAveragePool                                |                        |1                               |                                |
|com.microsoft           |QLinearLeakyRelu                                        |1                       |1                               |                                |
|com.microsoft           |QLinearMul                                              |1                       |1                               |                                |
|com.microsoft           |QLinearReduceMean                                       |                        |1                               |                                |
|com.microsoft           |QLinearSigmoid                                          |1                       |1                               |                                |
|com.microsoft           |QLinearSoftmax                                          |                        |1                               |                                |
|com.microsoft           |QLinearWhere                                            |                        |1                               |                                |
|com.microsoft           |QMoE                                                    |                        |1                               |                                |
|com.microsoft           |QOrderedAttention                                       |                        |1                               |                                |
|com.microsoft           |QOrderedGelu                                            |                        |1                               |                                |
|com.microsoft           |QOrderedLayerNormalization                              |                        |1                               |                                |
|com.microsoft           |QOrderedLongformerAttention                             |                        |1                               |                                |
|com.microsoft           |QOrderedMatMul                                          |                        |1                               |                                |
|com.microsoft           |QuantizeBFP                                             |                        |1                               |                                |
|com.microsoft           |QuantizeLinear                                          |1                       |1                               |                                |
|com.microsoft           |QuantizeWithOrder                                       |                        |1                               |                                |
|com.microsoft           |QuickGelu                                               |1                       |1                               |                                |
|com.microsoft           |Range                                                   |1                       |1                               |                                |
|com.microsoft           |ReduceSumInteger                                        |                        |1                               |                                |
|com.microsoft           |RelativePositionBias                                    |                        |1                               |                                |
|com.microsoft           |RemovePadding                                           |                        |1                               |                                |
|com.microsoft           |RestorePadding                                          |                        |1                               |                                |
|com.microsoft           |Rfft                                                    |                        |1                               |                                |
|com.microsoft           |RotaryEmbedding                                         |                        |1                               |                                |
|com.microsoft           |SampleOp                                                |                        |1                               |                                |
|com.microsoft           |Sampling                                                |                        |1                               |                                |
|com.microsoft           |SimplifiedLayerNormalization                            |1                       |1                               |                                |
|com.microsoft           |SkipGroupNorm                                           |                        |1                               |                                |
|com.microsoft           |SkipLayerNormalization                                  |1                       |1                               |                                |
|com.microsoft           |SkipSimplifiedLayerNormalization                        |1                       |1                               |                                |
|com.microsoft           |Snpe                                                    |                        |1                               |                                |
|com.microsoft           |SparseAttention                                         |                        |1                               |                                |
|com.microsoft           |SparseToDenseMatMul                                     |                        |1                               |                                |
|com.microsoft           |Tokenizer                                               |                        |1                               |                                |
|com.microsoft           |TorchEmbedding                                          |                        |1                               |                                |
|com.microsoft           |TransposeMatMul                                         |                        |1                               |                                |
|com.microsoft           |Trilu                                                   |1                       |1                               |                                |
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
