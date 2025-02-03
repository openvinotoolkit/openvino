# ONNX Operations Supported by OpenVINO ONNX Frontend

Here is a table of operations available by the ONNX Frontend.

OpenVINO provides support for operations of Default Opset (empty in table below), other operations are supported by community.

| Domain                 | Operation Name                                         | Since Version          | Limitation                     |
|------------------------|--------------------------------------------------------|------------------------|--------------------------------|
|                        |ATen                                                    |1                       |                                |
|                        |Abs                                                     |13, 6, 1                |                                |
|                        |Acos                                                    |1                       |                                |
|                        |Acosh                                                   |1                       |                                |
|                        |Add                                                     |14, 13, 7, 6, 1         |                                |
|                        |Affine                                                  |1                       |                                |
|                        |And                                                     |7, 1                    |                                |
|                        |ArgMax                                                  |12, 1                   |                                |
|                        |ArgMin                                                  |12, 1                   |                                |
|                        |Asin                                                    |1                       |                                |
|                        |Asinh                                                   |1                       |                                |
|                        |Atan                                                    |1                       |                                |
|                        |Atanh                                                   |1                       |                                |
|                        |AveragePool                                             |1                       |                                |
|                        |BatchNormalization                                      |14, 7, 1                |                                |
|                        |BitShift                                                |1                       |                                |
|                        |BitwiseAnd                                              |1                       |                                |
|                        |BitwiseNot                                              |1                       |                                |
|                        |BitwiseOr                                               |1                       |                                |
|                        |BitwiseXor                                              |1                       |                                |
|                        |BlackmanWindow                                          |1                       |                                |
|                        |Cast                                                    |1                       |                                |
|                        |CastLike                                                |1                       |                                |
|                        |Ceil                                                    |1                       |                                |
|                        |Celu                                                    |1                       |                                |
|                        |Clip                                                    |11, 1                   |                                |
|                        |Compress                                                |1                       |                                |
|                        |Concat                                                  |1                       |                                |
|                        |Constant                                                |13, 1                   |                                |
|                        |ConstantFill                                            |1                       |                                |
|                        |ConstantOfShape                                         |1                       |                                |
|                        |Conv                                                    |1                       |                                |
|                        |ConvInteger                                             |1                       |                                |
|                        |ConvTranspose                                           |1                       |                                |
|                        |Cos                                                     |1                       |                                |
|                        |Cosh                                                    |1                       |                                |
|                        |Crop                                                    |1                       |                                |
|                        |CumSum                                                  |1                       |                                |
|                        |DFT                                                     |1                       |                                |
|                        |DepthToSpace                                            |1                       |                                |
|                        |DequantizeLinear                                        |21, 19, 13, 1           |                                |
|                        |Div                                                     |7, 1                    |                                |
|                        |Dropout                                                 |12, 7, 1                |                                |
|                        |DynamicQuantizeLinear                                   |1                       |                                |
|                        |Einsum                                                  |1                       |                                |
|                        |Elu                                                     |1                       |                                |
|                        |Equal                                                   |1                       |                                |
|                        |Erf                                                     |1                       |                                |
|                        |Exp                                                     |1                       |                                |
|                        |Expand                                                  |1                       |                                |
|                        |EyeLike                                                 |1                       |                                |
|                        |Flatten                                                 |1                       |                                |
|                        |Floor                                                   |1                       |                                |
|                        |GRU                                                     |1                       |                                |
|                        |Gather                                                  |1                       |                                |
|                        |GatherElements                                          |1                       |                                |
|                        |GatherND                                                |1                       |                                |
|                        |Gelu                                                    |1                       |                                |
|                        |Gemm                                                    |6, 1                    |                                |
|                        |GlobalAveragePool                                       |1                       |                                |
|                        |GlobalLpPool                                            |1                       |                                |
|                        |GlobalMaxPool                                           |1                       |                                |
|                        |Greater                                                 |1                       |                                |
|                        |GreaterOrEqual                                          |16, 1                   |                                |
|                        |GridSample                                              |1                       |                                |
|                        |GroupNormalization                                      |1                       |                                |
|                        |HammingWindow                                           |1                       |                                |
|                        |HannWindow                                              |1                       |                                |
|                        |HardSigmoid                                             |1                       |                                |
|                        |HardSwish                                               |1                       |                                |
|                        |Hardmax                                                 |13, 1                   |                                |
|                        |Identity                                                |1                       |                                |
|                        |If                                                      |1                       |                                |
|                        |ImageScaler                                             |1                       |                                |
|                        |InstanceNormalization                                   |1                       |                                |
|                        |IsFinite                                                |1                       |                                |
|                        |IsInf                                                   |1                       |                                |
|                        |IsNaN                                                   |1                       |                                |
|                        |LRN                                                     |1                       |                                |
|                        |LSTM                                                    |1                       |                                |
|                        |LayerNormalization                                      |1                       |                                |
|                        |LeakyRelu                                               |1                       |                                |
|                        |Less                                                    |1                       |                                |
|                        |LessOrEqual                                             |16, 1                   |                                |
|                        |Log                                                     |1                       |                                |
|                        |LogSoftmax                                              |13, 1                   |                                |
|                        |Loop                                                    |1                       |                                |
|                        |LpNormalization                                         |1                       |                                |
|                        |MatMul                                                  |1                       |                                |
|                        |MatMulInteger                                           |1                       |                                |
|                        |Max                                                     |8, 1                    |                                |
|                        |MaxPool                                                 |8, 1                    |                                |
|                        |MaxRoiPool                                              |1                       |                                |
|                        |Mean                                                    |1                       |                                |
|                        |MeanVarianceNormalization                               |9, 1                    |                                |
|                        |Min                                                     |8, 1                    |                                |
|                        |Mish                                                    |1                       |                                |
|                        |Mod                                                     |1                       |                                |
|                        |Mul                                                     |7, 1                    |                                |
|                        |Multinomial                                             |1                       |                                |
|                        |Neg                                                     |1                       |                                |
|                        |NonMaxSuppression                                       |1                       |                                |
|                        |NonZero                                                 |1                       |                                |
|                        |Not                                                     |1                       |                                |
|                        |OneHot                                                  |1                       |                                |
|                        |Or                                                      |1                       |                                |
|                        |PRelu                                                   |1                       |                                |
|                        |Pad                                                     |11, 1                   |                                |
|                        |Pow                                                     |1                       |                                |
|                        |QLinearConv                                             |1                       |                                |
|                        |QLinearMatMul                                           |1                       |                                |
|                        |QuantizeLinear                                          |13, 1                   |                                |
|                        |RNN                                                     |1                       |                                |
|                        |RandomNormal                                            |1                       |                                |
|                        |RandomNormalLike                                        |1                       |                                |
|                        |RandomUniform                                           |1                       |                                |
|                        |RandomUniformLike                                       |1                       |                                |
|                        |Range                                                   |1                       |                                |
|                        |Reciprocal                                              |1                       |                                |
|                        |ReduceL1                                                |18, 1                   |                                |
|                        |ReduceL2                                                |18, 13, 1               |                                |
|                        |ReduceLogSum                                            |18, 1                   |                                |
|                        |ReduceLogSumExp                                         |18, 13, 1               |                                |
|                        |ReduceMax                                               |20, 18, 13, 1           |                                |
|                        |ReduceMean                                              |18, 13, 1               |                                |
|                        |ReduceMin                                               |20, 18, 13, 1           |                                |
|                        |ReduceProd                                              |18, 13, 1               |                                |
|                        |ReduceSum                                               |13, 1                   |                                |
|                        |ReduceSumSquare                                         |18, 13, 1               |                                |
|                        |Relu                                                    |1                       |                                |
|                        |Reshape                                                 |1                       |                                |
|                        |Resize                                                  |11, 1                   |                                |
|                        |ReverseSequence                                         |1                       |                                |
|                        |RoiAlign                                                |16, 1                   |                                |
|                        |Round                                                   |1                       |                                |
|                        |STFT                                                    |1                       |                                |
|                        |Scan                                                    |9, 1                    |                                |
|                        |Scatter                                                 |1                       |                                |
|                        |ScatterElements                                         |1                       |                                |
|                        |ScatterND                                               |1                       |                                |
|                        |Selu                                                    |1                       |                                |
|                        |Shape                                                   |15, 1                   |                                |
|                        |Shrink                                                  |1                       |                                |
|                        |Sigmoid                                                 |1                       |                                |
|                        |Sign                                                    |1                       |                                |
|                        |Sin                                                     |1                       |                                |
|                        |Sinh                                                    |1                       |                                |
|                        |Size                                                    |1                       |                                |
|                        |Slice                                                   |10, 1                   |                                |
|                        |Softmax                                                 |13, 11, 1               |                                |
|                        |Softplus                                                |1                       |                                |
|                        |Softsign                                                |1                       |                                |
|                        |SpaceToDepth                                            |1                       |                                |
|                        |Split                                                   |13, 1                   |                                |
|                        |Sqrt                                                    |1                       |                                |
|                        |Squeeze                                                 |13, 1                   |                                |
|                        |Sub                                                     |7, 1                    |                                |
|                        |Sum                                                     |8, 1                    |                                |
|                        |Tan                                                     |1                       |                                |
|                        |Tanh                                                    |1                       |                                |
|                        |ThresholdedRelu                                         |1                       |                                |
|                        |Tile                                                    |1                       |                                |
|                        |TopK                                                    |11, 10, 1               |                                |
|                        |Transpose                                               |1                       |                                |
|                        |Trilu                                                   |1                       |                                |
|                        |Unique                                                  |1                       |                                |
|                        |Unsqueeze                                               |13, 1                   |                                |
|                        |Upsample                                                |9, 7, 1                 |                                |
|                        |Where                                                   |1                       |                                |
|                        |Xor                                                     |1                       |                                |
|org.pytorch.aten        |adaptive_avg_pool2d                                     |1                       |                                |
|mmdeploy                |MMCVRoIAlignRotated                                     |1                       |                                |
|mmdeploy                |NMSRotated                                              |1                       |                                |
|com.microsoft           |Attention                                               |1                       |                                |
|com.microsoft           |BiasGelu                                                |1                       |                                |
|com.microsoft           |DequantizeLinear                                        |1                       |                                |
|com.microsoft           |DynamicQuantizeMatMul                                   |1                       |                                |
|com.microsoft           |EmbedLayerNormalization                                 |1                       |                                |
|com.microsoft           |FusedConv                                               |1                       |                                |
|com.microsoft           |FusedGemm                                               |1                       |                                |
|com.microsoft           |FusedMatMul                                             |1                       |                                |
|com.microsoft           |GatherND                                                |1                       |                                |
|com.microsoft           |Gelu                                                    |1                       |                                |
|com.microsoft           |MatMulIntegerToFloat                                    |1                       |                                |
|com.microsoft           |MatMulNBits                                             |1                       |                                |
|com.microsoft           |Pad                                                     |1                       |                                |
|com.microsoft           |QLinearAdd                                              |1                       |                                |
|com.microsoft           |QLinearLeakyRelu                                        |1                       |                                |
|com.microsoft           |QLinearMul                                              |1                       |                                |
|com.microsoft           |QLinearSigmoid                                          |1                       |                                |
|com.microsoft           |QuantizeLinear                                          |1                       |                                |
|com.microsoft           |QuickGelu                                               |1                       |                                |
|com.microsoft           |Range                                                   |1                       |                                |
|com.microsoft           |SimplifiedLayerNormalization                            |1                       |                                |
|com.microsoft           |SkipLayerNormalization                                  |1                       |                                |
|com.microsoft           |SkipSimplifiedLayerNormalization                        |1                       |                                |
|com.microsoft           |Trilu                                                   |1                       |                                |
|org.openvinotoolkit     |DeformableConv2D                                        |1                       |                                |
|org.openvinotoolkit     |DetectionOutput                                         |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronDetectionOutput                    |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronGenerateProposalsSingleImage       |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronGroupNorm                          |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronPriorGridGenerator                 |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronROIFeatureExtractor                |1                       |                                |
|org.openvinotoolkit     |ExperimentalDetectronTopKROIs                           |1                       |                                |
|org.openvinotoolkit     |FakeQuantize                                            |1                       |                                |
|org.openvinotoolkit     |GenerateProposals                                       |1                       |                                |
|org.openvinotoolkit     |GroupNorm                                               |1                       |                                |
|org.openvinotoolkit     |Normalize                                               |1                       |                                |
|org.openvinotoolkit     |PriorBox                                                |1                       |                                |
|org.openvinotoolkit     |PriorBoxClustered                                       |1                       |                                |
|org.openvinotoolkit     |Swish                                                   |1                       |                                |
