.. {#openvino_resources_supported_operations}

Supported Operations - by Inference Devices
===========================================


This page lists operations supported by OpenVINO inference devices. The table presents general information,
for a more detailed and most recent listing of operations that are implemented and tested:

.. button-link:: _static/download/operation_conformance_table_files/opset_report_omz_static.html
   :color: primary
   :outline:

   See the full conformance report table


=================================  ===============  ==============  ================  ==================
 Operations                         CPU (x86)        GPU             GNA               CPU (Arm®)
=================================  ===============  ==============  ================  ==================  
 Abs                                Supported**      Supported       Not Supported     Supported       
 Acos                               Supported**      Supported       Not Supported     Supported****
 Acosh                              Supported**      Supported       Not Supported     Supported****
 Activation-Clamp                   Supported***     Supported       Supported         Supported       
 Activation-ELU                     Supported***     Supported       Not Supported     Supported       
 Activation-Exp                     Supported***     Supported       Supported         Supported       
 Activation-Leaky ReLU              Supported***     Supported       Supported         Not Supported   
 Activation-Not                     Supported***     Supported       Not Supported     Not Supported   
 Activation-PReLU                   Supported***     Supported       Not Supported     Supported       
 Activation-ReLU                    Supported***     Supported       Supported         Supported       
 Activation-ReLU6                   Supported***     Supported       Not Supported     Not Supported   
 Activation-Sigmoid/Logistic        Supported***     Supported       Supported         Supported       
 Activation-TanH                    Supported***     Supported       Supported         Supported       
 ArgMax                             Supported**      Supported       Not Supported     Not Supported   
 Asin                               Supported**      Supported       Not Supported     Supported****
 Asinh                              Supported**      Supported       Not Supported     Supported****
 Atan                               Supported**      Supported       Not Supported     Supported****
 Atanh                              Supported**      Supported       Not Supported     Supported****
 BatchNormalization                 Supported        Supported       Not Supported     Supported       
 BinaryConvolution                  Supported        Supported       Not Supported     Not Supported   
 Broadcast                          Supported**      Supported       Not Supported     Supported       
 Ceil                               Supported**      Supported       Not Supported     Supported       
 Concat                             Supported***     Supported       Supported         Supported       
 Const                              Supported        Supported       Supported         Supported       
 Convolution-Dilated                Supported        Supported       Not Supported     Supported       
 Convolution-Dilated 3D             Supported        Supported       Not Supported     Not Supported   
 Convolution-Grouped                Supported        Supported       Not Supported     Supported       
 Convolution-Grouped 3D             Supported        Supported       Not Supported     Not Supported   
 Convolution-Ordinary               Supported        Supported       Supported*        Supported       
 Convolution-Ordinary 3D            Supported        Supported       Not Supported     Not Supported   
 Cos                                Supported**      Supported       Not Supported     Supported****
 Cosh                               Supported**      Supported       Not Supported     Supported****
 Crop                               Supported        Supported       Supported         Not Supported   
 CTCGreedyDecoder                   Supported**      Supported**     Not Supported     Supported****
 Deconvolution                      Supported        Supported       Not Supported     Not Supported   
 Deconvolution 3D                   Supported        Supported       Not Supported     Not Supported   
 DeformableConvolution              Supported        Supported       Not Supported     Not Supported   
 DepthToSpace                       Supported**      Supported       Not Supported     Supported*     
 DetectionOutput                    Supported**      Supported       Not Supported     Supported****
 Eltwise-And                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Add                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Div                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Equal                      Supported***     Supported       Not Supported     Supported*     
 Eltwise-FloorMod                   Supported***     Supported       Not Supported     Supported****
 Eltwise-Greater                    Supported***     Supported       Not Supported     Supported       
 Eltwise-GreaterEqual               Supported***     Supported       Not Supported     Supported       
 Eltwise-Less                       Supported***     Supported       Not Supported     Supported*     
 Eltwise-LessEqual                  Supported***     Supported       Not Supported     Supported*     
 Eltwise-LogicalAnd                 Supported***     Supported       Not Supported     Supported       
 Eltwise-LogicalOr                  Supported***     Supported       Not Supported     Supported       
 Eltwise-LogicalXor                 Supported***     Supported       Not Supported     Supported       
 Eltwise-Max                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Min                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Mul                        Supported***     Supported       Supported         Supported       
 Eltwise-NotEqual                   Supported***     Supported       Not Supported     Supported*     
 Eltwise-Pow                        Supported***     Supported       Not Supported     Supported       
 Eltwise-Prod                       Supported***     Supported       Supported         Not Supported   
 Eltwise-SquaredDiff                Supported***     Supported       Not Supported     Supported       
 Eltwise-Sub                        Supported***     Supported       Supported         Supported       
 Eltwise-Sum                        Supported***     Supported       Supported         Supported****
 Erf                                Supported**      Supported       Not Supported     Supported****
 Exp                                Supported        Supported       Supported         Supported       
 FakeQuantize                       Supported        Not Supported   Not Supported     Supported*     
 Fill                               Supported**      Not Supported   Not Supported     Not Supported   
 Flatten                            Supported        Supported       Not Supported     Not Supported   
 Floor                              Supported**      Supported       Not Supported     Supported       
 FullyConnected (Inner Product)     Supported***     Supported       Supported         Supported       
 Gather                             Supported**      Supported       Not Supported     Supported*     
 GatherTree                         Supported**      Not Supported   Not Supported     Supported****
 Gemm                               Supported        Supported       Not Supported     Not Supported   
 GRN                                Supported**      Supported**     Not Supported     Supported       
 HardSigmoid                        Supported**      Supported       Not Supported     Supported****
 Interp                             Supported**      Supported**     Not Supported     Supported*     
 Log                                Supported**      Supported       Supported         Supported       
 LRN (Norm)                         Supported        Supported       Not Supported     Supported*     
 LSTMCell                           Supported        Supported       Supported         Supported       
 GRUCell                            Supported        Supported       Supported         Supported       
 RNNCell                            Supported        Supported       Not Supported     Supported       
 LSTMSequence                       Supported        Supported       Supported         Supported****
 GRUSequence                        Supported        Supported       Supported         Supported****
 RNNSequence                        Supported        Supported       Not Supported     Supported****
 LogSoftmax                         Supported**      Supported       Not Supported     Supported       
 Memory                             Supported        Not Supported   Supported         Not Supported   
 MVN                                Supported**      Supported       Not Supported     Supported*     
 Neg                                Supported**      Supported       Not Supported     Supported       
 NonMaxSuppression                  Supported**      Not Supported   Not Supported     Supported****
 Normalize                          Supported**      Supported       Not Supported     Supported*     
 OneHot                             Supported**      Supported       Not Supported     Supported****
 Pad                                Supported**      Supported       Not Supported     Supported*     
 Permute                            Supported        Supported       Supported*        Not Supported   
 Pooling(AVG,MAX)                   Supported        Supported       Supported         Supported       
 Pooling(AVG,MAX) 3D                Supported        Supported       Not Supported     Supported*     
 Power                              Supported**      Supported       Supported*        Supported       
 PowerFile                          Supported**      Not Supported   Not Supported     Not Supported   
 PriorBox                           Supported**      Supported       Not Supported     Supported       
 PriorBoxClustered                  Supported**      Supported**     Not Supported     Supported       
 Proposal                           Supported**      Supported       Not Supported     Supported****
 PSROIPooling                       Supported**      Supported       Not Supported     Supported****
 Range                              Supported**      Not Supported   Not Supported     Not Supported   
 Reciprocal                         Supported**      Supported       Not Supported     Not Supported   
 ReduceAnd                          Supported**      Supported       Not Supported     Supported****
 ReduceL1                           Supported**      Supported       Not Supported     Supported       
 ReduceL2                           Supported**      Supported       Not Supported     Supported       
 ReduceLogSum                       Supported**      Supported       Not Supported     Supported       
 ReduceLogSumExp                    Supported**      Supported       Not Supported     Not Supported   
 ReduceMax                          Supported**      Supported       Not Supported     Supported       
 ReduceMean                         Supported**      Supported       Not Supported     Supported       
 ReduceMin                          Supported**      Supported       Not Supported     Supported       
 ReduceOr                           Supported**      Supported       Not Supported     Supported****
 ReduceProd                         Supported**      Supported       Not Supported     Supported       
 ReduceSum                          Supported**      Supported       Not Supported     Supported       
 ReduceSumSquare                    Supported**      Supported       Not Supported     Not Supported   
 RegionYolo                         Supported**      Supported       Not Supported     Supported****
 ReorgYolo                          Supported**      Supported       Not Supported     Supported       
 Resample                           Supported**      Supported       Not Supported     Not Supported   
 Reshape                            Supported***     Supported       Supported         Supported       
 ReverseSequence                    Supported**      Supported       Not Supported     Supported****
 RNN                                Supported        Not Supported   Not Supported     Supported       
 ROIPooling                         Supported        Supported*      Not Supported     Supported****
 ScaleShift                         Supported***     Supported       Supported         Not Supported   
 ScatterUpdate                      Supported**      Not Supported   Not Supported     Not Supported   
 Select                             Supported        Supported       Not Supported     Supported       
 Selu                               Supported**      Supported       Not Supported     Supported****
 ShuffleChannels                    Supported**      Supported       Not Supported     Supported       
 Sign                               Supported**      Supported       Not Supported     Supported       
 Sin                                Supported**      Supported       Not Supported     Supported       
 Sinh                               Supported**      Supported       Not Supported     Supported****
 SimplerNMS                         Supported**      Supported       Not Supported     Not Supported   
 Slice                              Supported***     Supported       Supported         Not Supported   
 SoftMax                            Supported***     Supported       Not Supported     Supported       
 Softplus                           Supported**      Supported       Not Supported     Supported       
 Softsign                           Supported**      Supported       Supported         Not Supported   
 SpaceToDepth                       Supported**      Not Supported   Not Supported     Supported*     
 SpatialTransformer                 Supported**      Not Supported   Not Supported     Not Supported   
 Split                              Supported***     Supported       Supported         Supported       
 Squeeze                            Supported**      Supported       Supported         Supported       
 StridedSlice                       Supported**      Supported       Not Supported     Supported*     
 Tan                                Supported**      Supported       Not Supported     Supported****
 TensorIterator                     Supported        Not Supported   Supported         Supported       
 Tile                               Supported***     Supported**     Not Supported     Supported       
 TopK                               Supported**      Supported       Not Supported     Supported****
 Unpooling                          Not Supported    Supported       Not Supported     Not Supported   
 Unsqueeze                          Supported**      Supported       Supported         Supported       
 Upsampling                         Not Supported    Supported       Not Supported     Not Supported   
=================================  ===============  ==============  ================  ==================

|   `*` - support is limited to the specific parameters. Refer to "Known Layer Limitations" section for the device :doc:`from the list of supported <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`.
|   `**` - support is implemented via :doc:`Extensibility mechanism <openvino_docs_Extensibility_UG_Intro>`.
|   `***` - supports NCDHW layout.
|   `****` - support is implemented via runtime reference.



