.. {#openvino_resources_supported_operations}

Supported Operations - by Inference Devices
===========================================


This page presents operations supported by OpenVINO inference devices. The table presents general information,
for a more detailed and most recent listing of operations that are implemented and tested:


.. button-link:: ../../_static/download/conformance_reports/opset_report_omz_static.html
   :color: primary
   :outline:

   See the full conformance report table (static)

.. button-link:: ../../_static/download/conformance_reports/opset_report_omz_dynamic.html
   :color: primary
   :outline:

   See the full conformance report table (dynamic)



=================================  ===============  ==============  ==================
 Operations                         CPU (x86)        GPU             CPU (Arm®)
=================================  ===============  ==============  ==================
 Abs                                Supported**      Supported       Supported
 Acos                               Supported**      Supported       Supported****
 Acosh                              Supported**      Supported       Supported****
 Activation-Clamp                   Supported***     Supported       Supported
 Activation-ELU                     Supported***     Supported       Supported
 Activation-Exp                     Supported***     Supported       Supported
 Activation-Leaky ReLU              Supported***     Supported       Not Supported
 Activation-Not                     Supported***     Supported       Not Supported
 Activation-PReLU                   Supported***     Supported       Supported
 Activation-ReLU                    Supported***     Supported       Supported
 Activation-ReLU6                   Supported***     Supported       Not Supported
 Activation-Sigmoid/Logistic        Supported***     Supported       Supported
 Activation-TanH                    Supported***     Supported       Supported
 ArgMax                             Supported**      Supported       Not Supported
 Asin                               Supported**      Supported       Supported****
 Asinh                              Supported**      Supported       Supported****
 Atan                               Supported**      Supported       Supported****
 Atanh                              Supported**      Supported       Supported****
 BatchNormalization                 Supported        Supported       Supported
 BinaryConvolution                  Supported        Supported       Not Supported
 Broadcast                          Supported**      Supported       Supported
 Ceil                               Supported**      Supported       Supported
 Concat                             Supported***     Supported       Supported
 Const                              Supported        Supported       Supported
 Convolution-Dilated                Supported        Supported       Supported
 Convolution-Dilated 3D             Supported        Supported       Not Supported
 Convolution-Grouped                Supported        Supported       Supported
 Convolution-Grouped 3D             Supported        Supported       Not Supported
 Convolution-Ordinary               Supported        Supported       Supported
 Convolution-Ordinary 3D            Supported        Supported       Not Supported
 Cos                                Supported**      Supported       Supported****
 Cosh                               Supported**      Supported       Supported****
 Crop                               Supported        Supported       Not Supported
 CTCGreedyDecoder                   Supported**      Supported**     Supported****
 Deconvolution                      Supported        Supported       Not Supported
 Deconvolution 3D                   Supported        Supported       Not Supported
 DeformableConvolution              Supported        Supported       Not Supported
 DepthToSpace                       Supported**      Supported       Supported*
 DetectionOutput                    Supported**      Supported       Supported****
 Eltwise-And                        Supported***     Supported       Supported
 Eltwise-Add                        Supported***     Supported       Supported
 Eltwise-Div                        Supported***     Supported       Supported
 Eltwise-Equal                      Supported***     Supported       Supported*
 Eltwise-FloorMod                   Supported***     Supported       Supported****
 Eltwise-Greater                    Supported***     Supported       Supported
 Eltwise-GreaterEqual               Supported***     Supported       Supported
 Eltwise-Less                       Supported***     Supported       Supported*
 Eltwise-LessEqual                  Supported***     Supported       Supported*
 Eltwise-LogicalAnd                 Supported***     Supported       Supported
 Eltwise-LogicalOr                  Supported***     Supported       Supported
 Eltwise-LogicalXor                 Supported***     Supported       Supported
 Eltwise-Max                        Supported***     Supported       Supported
 Eltwise-Min                        Supported***     Supported       Supported
 Eltwise-Mul                        Supported***     Supported       Supported
 Eltwise-NotEqual                   Supported***     Supported       Supported*
 Eltwise-Pow                        Supported***     Supported       Supported
 Eltwise-Prod                       Supported***     Supported       Not Supported
 Eltwise-SquaredDiff                Supported***     Supported       Supported
 Eltwise-Sub                        Supported***     Supported       Supported
 Eltwise-Sum                        Supported***     Supported       Supported****
 Erf                                Supported**      Supported       Supported****
 Exp                                Supported        Supported       Supported
 FakeQuantize                       Supported        Not Supported   Supported*
 Fill                               Supported**      Not Supported   Not Supported
 Flatten                            Supported        Supported       Not Supported
 Floor                              Supported**      Supported       Supported
 FullyConnected (Inner Product)     Supported***     Supported       Supported
 Gather                             Supported**      Supported       Supported*
 GatherTree                         Supported**      Not Supported   Supported****
 Gemm                               Supported        Supported       Not Supported
 GRN                                Supported**      Supported**     Supported
 HardSigmoid                        Supported**      Supported       Supported****
 Interp                             Supported**      Supported**     Supported*
 Log                                Supported**      Supported       Supported
 LRN (Norm)                         Supported        Supported       Supported*
 LSTMCell                           Supported        Supported       Supported
 GRUCell                            Supported        Supported       Supported
 RNNCell                            Supported        Supported       Supported
 LSTMSequence                       Supported        Supported       Supported****
 GRUSequence                        Supported        Supported       Supported****
 RNNSequence                        Supported        Supported       Supported****
 LogSoftmax                         Supported**      Supported       Supported
 Memory                             Supported        Not Supported   Not Supported
 MVN                                Supported**      Supported       Supported*
 Neg                                Supported**      Supported       Supported
 NonMaxSuppression                  Supported**      Not Supported   Supported****
 Normalize                          Supported**      Supported       Supported*
 OneHot                             Supported**      Supported       Supported****
 Pad                                Supported**      Supported       Supported*
 Permute                            Supported        Supported       Not Supported
 Pooling(AVG,MAX)                   Supported        Supported       Supported
 Pooling(AVG,MAX) 3D                Supported        Supported       Supported*
 Power                              Supported**      Supported       Supported
 PowerFile                          Supported**      Not Supported   Not Supported
 PriorBox                           Supported**      Supported       Supported
 PriorBoxClustered                  Supported**      Supported**     Supported
 Proposal                           Supported**      Supported       Supported****
 PSROIPooling                       Supported**      Supported       Supported****
 Range                              Supported**      Not Supported   Not Supported
 Reciprocal                         Supported**      Supported       Not Supported
 ReduceAnd                          Supported**      Supported       Supported****
 ReduceL1                           Supported**      Supported       Supported
 ReduceL2                           Supported**      Supported       Supported
 ReduceLogSum                       Supported**      Supported       Supported
 ReduceLogSumExp                    Supported**      Supported       Not Supported
 ReduceMax                          Supported**      Supported       Supported
 ReduceMean                         Supported**      Supported       Supported
 ReduceMin                          Supported**      Supported       Supported
 ReduceOr                           Supported**      Supported       Supported****
 ReduceProd                         Supported**      Supported       Supported
 ReduceSum                          Supported**      Supported       Supported
 ReduceSumSquare                    Supported**      Supported       Not Supported
 RegionYolo                         Supported**      Supported       Supported****
 ReorgYolo                          Supported**      Supported       Supported
 Resample                           Supported**      Supported       Not Supported
 Reshape                            Supported***     Supported       Supported
 ReverseSequence                    Supported**      Supported       Supported****
 RNN                                Supported        Not Supported   Supported
 ROIPooling                         Supported        Supported*      Supported****
 ScaleShift                         Supported***     Supported       Not Supported
 ScatterUpdate                      Supported**      Not Supported   Not Supported
 Select                             Supported        Supported       Supported
 Selu                               Supported**      Supported       Supported****
 ShuffleChannels                    Supported**      Supported       Supported
 Sign                               Supported**      Supported       Supported
 Sin                                Supported**      Supported       Supported
 Sinh                               Supported**      Supported       Supported****
 SimplerNMS                         Supported**      Supported       Not Supported
 Slice                              Supported***     Supported       Not Supported
 SoftMax                            Supported***     Supported       Supported
 Softplus                           Supported**      Supported       Supported
 Softsign                           Supported**      Supported       Not Supported
 SpaceToDepth                       Supported**      Not Supported   Supported*
 SpatialTransformer                 Supported**      Not Supported   Not Supported
 Split                              Supported***     Supported       Supported
 Squeeze                            Supported**      Supported       Supported
 StridedSlice                       Supported**      Supported       Supported*
 Tan                                Supported**      Supported       Supported****
 TensorIterator                     Supported        Not Supported   Supported
 Tile                               Supported***     Supported**     Supported
 TopK                               Supported**      Supported       Supported****
 Unpooling                          Not Supported    Supported       Not Supported
 Unsqueeze                          Supported**      Supported       Supported
 Upsampling                         Not Supported    Supported       Not Supported
=================================  ===============  ==============  ==================

|   `*` - support is limited to the specific parameters.
|   `**` - support is implemented via :doc:`Extensibility mechanism <../../documentation/openvino-extensibility>`.
|   `***` - supports NCDHW layout.
|   `****` - support is implemented via runtime reference.



