# Supported Framework Layers {#openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers}


In this article, you can find lists of supported framework layers, divided by frameworks.

## Caffe Supported Layers


| Layer Name in Caffe | Limitations |
|:---------- | :----------|
| Axpy |  |
| BN |  |
| BatchNorm |  |
| Bias |  |
| Binarization (Intel experimental) |  |
| Concat |  |
| Convolution |  |
| ConvolutionBinary |  |
| Crop |  |
| Deconvolution |  |
| DetectionOutput |  |
| Dropout | Not needed for inference. |
| Eltwise |  |
| Flatten |  |
| GlobalInput |  |
| InnerProduct |  |
| Input |  |
| LRN |  |
| Normalize |  |
| Python | Supported only for the Python Proposal operation. |
| Permute |  |
| Pooling |  |
| Power |  |
| PReLU |  |
| PriorBox |  |
| PriorBoxClustered |  |
| Proposal |  |
| PSROIPooling |  |
| ROIPooling |  |
| RegionYolo |  |
| ReorgYolo |  |
| ReLU |  |
| Resample |  |
| Reshape |  |
| Scale |  |
| ShuffleChannel |  |
| Sigmoid |  |
| Slice |  |
| Softmax |  |
| Tile |  |


## MXNet Supported Symbols


| Symbol Name in MXNet| Limitations|
| :----------| :----------|
| _Plus |  |
| _contrib_arange_like |  |
| _contrib_box_nms |  |
| _contrib_DeformableConvolution |  |
| _contrib_DeformablePSROIPooling |  |
| _contrib_div_sqrt_dim |  |
| _contrib_MultiBoxDetection | The *`force_suppress`* = 1 is not supported, non-default variances are not supported. |
| _contrib_MultiBoxPrior |  |
| _contrib_Proposal |  |
| _copy | Not needed for inference |
| _div_scalar |  |
| _greater_scalar |  |
| _minus_scalar |  |
| _mul_scalar |  |
| _plus_scalar |  |
| _random_uniform | Operation provides sequence from uniform distribution, but exact values won't match. |
| _rnn_param_concat |  |
| _arange |  |
| _contrib_AdaptiveAvgPooling2D | Converted to the Average Pooling with fixed paddings. |
| _maximum |  |
| _minimum |  |
| _np_roll |  |
| _zeros |  |
| add_n |  |
| arccosh |  |
| arcsinh |  |
| arctanh |  |
| batch_dot |  |
| broadcast_add |  |
| broadcast_div |  |
| broadcast_mul |  |
| broadcast_sub |  |
| BlockGrad |  |
| cumsum |  |
| div_scalar |  |
| elementwise_sub |  |
| elemwise_add |  |
| elemwise_mul |  |
| elemwise_sub |  |
| exp |  |
| expand_dims |  |
| greater_scalar |  |
| max |  |
| minus_scalar |  |
| null | Not needed for inference. |
| LayerNorm | The *`output_mean_var`* = True is not supported. |
| repeat |  |
| rnn |  |
| rnn_param_concat |  |
| round |  |
| sigmoid |  |
| slice |  |
| SliceChannel |  |
| slice_axis |  |
| slice_channel |  |
| slice_like |  |
| softmax |  |
| stack |  |
| swapaxis |  |
| tile |  |
| transpose |  |
| zeros |  |
| Activation | Supported *`act_type`* = *`relu`*, *`sigmoid`*, *`softrelu`* or *`tanh`*. |
| BatchNorm |  |
| Concat |  |
| Convolution |  |
| Crop | The *`center_crop`* = 1 is not supported. |
| Custom | [Custom Layers in Model Optimizer.](customize_model_optimizer/Customize_Model_Optimizer.md) |
| Deconvolution |  |
| DeformableConvolution |  |
| DeformablePSROIPooling |  |
| Dropout | Not needed for inference. |
| ElementWiseSum |  |
| Embedding |  |
| Flatten |  |
| FullyConnected |  |
| InstanceNorm |  |
| L2Normalization | Only 4D input is supported. |
| LRN |  |
| LeakyReLU | Supported *`act_type`* = *`prelu`*, *`elu`*, *`leaky`*, *`gelu`*. |
| ones_like |  |
| Pad |  |
| Pooling |  |
| ROIPooling |  |
| ReLU |  |
| Reshape |  |
| ScaleShift |  |
| SoftmaxActivation |  |
| SoftmaxOutput |  |
| SoftSign |  |
| Take | The attribute *`mode`* is not supported. |
| Tile |  |
| UpSampling |  |
| Where |  |
| zeros_like |  |


## TensorFlow Supported Operations

Some of TensorFlow operations do not match any OpenVINO operations. Yet, they are still supported by Model Optimizer and can be used on constant propagation path. These layers are labeled *`Constant propagation`* in the table below:


| Operation Name in TensorFlow | Limitations|
| :----------| :----------|
| Abs |  |
| Acosh |  |
| Add |  |
| AddV2 |  |
| AddN |  |
| All |  |
| ArgMax |  |
| ArgMin |  |
| Asinh |  |
| Assert | Not needed for inference. |
| Assign | Not needed for inference. |
| AssignSub | Not needed for inference. |
| Atanh |  |
| AvgPool |  |
| AvgPoolV2 | Supported only for constant-foldable *`kernel_size`* and strides inputs. |
| AvgPool3D |  |
| BatchMatMul |  |
| BatchMatMulV2 |  |
| BatchToSpaceND |  |
| BiasAdd |  |
| BlockLSTM |  |
| Bucketize | CPU only. |
| BroadcastTo |  |
| Cast |  |
| Ceil |  |
| ClipByValue |  |
| Concat |  |
| ConcatV2 |  |
| Const |  |
| Conv2D |  |
| Conv2DBackpropInput |  |
| Conv3D |  |
| Conv3DBackpropInputV2 |  |
| Cos |  |
| Cosh |  |
| CropAndResize | The *`method`* = *`bilinear`* only. |
| CTCGreedyDecoder | Supported only with decoded indices output in a dense format. |
| CTCLoss | Supported only with decoded indices input in a dense format. |
| CumSum |  |
| DepthToSpace|  |
| DepthwiseConv2dNative|  |
| Einsum | Supported only with equation that does not contain repeated labels within a subscript. |
| Elu |  |
| EmptyTensorList | Supported only when it is part of a sub-graph of the special form. |
| Enter | Supported only when it is fused to the TensorIterator layer. |
| Equal |  |
| Erf |  |
| Exit | Supported only when it is fused to the TensorIterator layer. |
| Exp |  |
| ExpandDims |  |
| ExperimentalSparseWeightedSum | CPU only. |
| ExtractImagePatches |  |
| EuclideanNorm |  |
| FakeQuantWithMinMaxVars |  |
| FakeQuantWithMinMaxVarsPerChannel |  |
| FFT | Supported only when it is part of a sub-graph of the special form. |
| FFT2D | Supported only when it is part of a sub-graph of the special form. |
| FFT3D | Supported only when it is part of a sub-graph of the special form. |
| FIFOQueueV2 | Supported only when it is part of a sub-graph of the special form. |
| Fill |  |
| Floor |  |
| FloorDiv |  |
| FloorMod |  |
| FusedBatchNorm |  |
| FusedBatchNormV2 |  |
| FusedBatchNormV3 |  |
| Gather |  |
| GatherNd |  |
| GatherTree |  |
| GatherV2 |  |
| Greater |  |
| GreaterEqual |  |
| Identity | Not needed for shape inference. |
| IdentityN |  |
| IFFT | Supported only when it is part of a sub-graph of the special form. |
| IFFT2D | Supported only when it is part of a sub-graph of the special form. |
| IFFT3D | Supported only when it is part of a sub-graph of the special form. |
| IteratorGetNext | Supported only when it is part of a sub-graph of the special form. |
| LRN |  |
| LeakyRelu |  |
| Less |  |
| LessEqual |  |
| Log |  |
| Log1p |  |
| LogicalAnd |  |
| LogicalOr |  |
| LogicalNot |  |
| LogSoftmax |  |
| LookupTableInsertV2 | Supported only when it is part of a sub-graph of the special form. |
| LoopCond | Supported only when it is fused to the TensorIterator layer. |
| MatMul |  |
| Max |  |
| MaxPool |  |
| MaxPoolV2 | Supported only for constant-foldable *`kernel_size`* and strides inputs. |
| MaxPool3D |  |
| Maximum |  |
| Mean |  |
| Merge | Supported only when it is fused to the TensorIterator layer. |
| Min |  |
| Minimum |  |
| MirrorPad |  |
| Mod |  |
| Mul |  |
| Neg |  |
| NextIteration | Supported only when it is fused to the TensorIterator layer. |
| NonMaxSuppressionV2 |  |
| NonMaxSuppressionV3 |  |
| NonMaxSuppressionV4 |  |
| NonMaxSuppressionV5 |  |
| NotEqual |  |
| NoOp |  |
| OneHot |  |
| Pack |  |
| Pad |  |
| PadV2 |  |
| Placeholder |  |
| PlaceholderWithDefault |  |
| Prod |  |
| QueueDequeue | Supported only when it is part of a sub-graph of the special form. |
| QueueDequeueUpToV2 | Supported only when it is part of a sub-graph of the special form. |
| QueueDequeueV2 | Supported only when it is part of a sub-graph of the special form. |
| RandomUniform |  |
| RandomUniformInt |  |
| Range |  |
| Rank |  |
| RealDiv |  |
| Reciprocal |  |
| Relu |  |
| Relu6 |  |
| Reshape |  |
| ResizeBilinear |  |
| ResizeNearestNeighbor |  |
| ResourceGather|  |
| ReverseSequence |  |
| ReverseV2 | Supported only when it can be converted to the ReverseSequence operation. |
| Roll |  |
| Round |  |
| Pow |  |
| Rsqrt |  |
| ScatterNd |  |
| Select |  |
| SelectV2 |  |
| Shape |  |
| Sigmoid |  |
| Sin |  |
| Sinh |  |
| Size |  |
| Slice |  |
| Softmax |  |
| Softplus |  |
| Softsign |  |
| SpaceToBatchND |  |
| SpaceToDepth |  |
| SparseFillEmptyRows | Supported only when it is part of a sub-graph of the special form. |
| SparseReshape | Supported only when it is part of a sub-graph of the special form. |
| SparseSegmentSum | Supported only when it is part of a sub-graph of the special form. |
| SparseSegmentMean | Supported only when it is part of a sub-graph of the special form. |
| SparseToDense | CPU only |
| Split |  |
| SplitV |  |
| Sqrt |  |
| Square |  |
| SquaredDifference |  |
| Square|  |
| Squeeze | Cases in which squeeze axis is not specified are not supported. |
| StatelessWhile |  |
| StopGradient | Not needed for shape inference. |
| StridedSlice | Supported only for constant-foldable *`begin`*, *`end`*, and *`strides`* inputs. |
| Sub |  |
| Sum |  |
| Swish |  |
| swish_f32 |  |
| Switch | Control flow propagation. |
| Tan |  |
| Tanh |  |
| TensorArrayGatherV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorArrayReadV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorArrayScatterV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorArraySizeV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorArrayV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorArrayWriteV3 | Supported only when it is fused to the TensorIterator layer. |
| TensorListPushBack | Supported only when it is part of a sub-graph of the special form. |
| Tile |  |
| TopkV2 |  |
| Transpose |  |
| Unpack |  |
| Variable |  |
| VariableV2 |  |
| Where | Supported only when it is part of a sub-graph of the special form. |
| ZerosLike |  |


## TensorFlow 2 Keras Supported Operations


| Operation Name in TensorFlow 2 Keras | Limitations|
| :----------| :----------|
| ActivityRegularization |  |
| Add |  |
| AdditiveAttention |  |
| AlphaDropout |  |
| Attention |  |
| Average |  |
| AveragePooling1D |  |
| AveragePooling2D |  |
| AveragePooling3D |  |
| BatchNormalization |  |
| Bidirectional |  |
| Concatenate |  |
| Conv1D |  |
| Conv1DTranspose | Not supported if dilation is not equal to 1. |
| Conv2D |  |
| Conv2DTranspose |  |
| Conv3D |  |
| Conv3DTranspose |  |
| Cropping1D |  |
| Cropping2D |  |
| Cropping3D |  |
| Dense |  |
| DenseFeatures | Not supported for categorical and crossed features. |
| DepthwiseConv2D |  |
| Dot |  |
| Dropout |  |
| ELU |  |
| Embedding |  |
| Flatten |  |
| GRU |  |
| GRUCell |  |
| GaussianDropout |  |
| GaussianNoise |  |
| GlobalAveragePooling1D |  |
| GlobalAveragePooling2D |  |
| GlobalAveragePooling3D |  |
| GlobalMaxPool1D |  |
| GlobalMaxPool2D |  |
| GlobalMaxPool3D |  |
| LSTM |  |
| LSTMCell |  |
| Lambda |  |
| LayerNormalization |  |
| LeakyReLU |  |
| LocallyConnected1D |  |
| LocallyConnected2D |  |
| MaxPool1D |  |
| MaxPool2D |  |
| MaxPool3D |  |
| Maximum |  |
| Minimum |  |
| Multiply |  |
| PReLU |  |
| Permute |  |
| RNN | Not supported for some custom cells. |
| ReLU |  |
| RepeatVector |  |
| Reshape |  |
| Roll |  |
| SeparableConv1D |  |
| SeparableConv2D |  |
| SimpleRNN |  |
| SimpleRNNCell |  |
| Softmax |  |
| SpatialDropout1D |  |
| SpatialDropout2D |  |
| SpatialDropout3D |  |
| StackedRNNCells |  |
| Subtract |  |
| ThresholdedReLU |  |
| TimeDistributed |  |
| UpSampling1D |  |
| UpSampling2D |  |
| UpSampling3D |  |
| ZeroPadding1D |  |
| ZeroPadding2D |  |
| ZeroPadding3D |  |

## Kaldi Supported Layers


| Symbol Name in Kaldi| Limitations|
| :----------| :----------|
| addshift |  |
| affinecomponent |  |
| affinecomponentpreconditionedonline |  |
| affinetransform |  |
| backproptruncationcomponent |  |
| batchnormcomponent |  |
| clipgradientcomponent | Not needed for inference. |
| concat |  |
| convolutional1dcomponent |  |
| convolutionalcomponent |  |
| copy |  |
| dropoutmaskcomponent |  |
| elementwiseproductcomponent |  |
| fixedaffinecomponent |  |
| fixedbiascomponent |  |
| fixedscalecomponent |  |
| generaldropoutcomponent| Not needed for inference. |
| linearcomponent |  |
| logsoftmaxcomponent |  |
| lstmnonlinearitycomponent |  |
| lstmprojected |  |
| lstmprojectedstreams |  |
| maxpoolingcomponent |  |
| naturalgradientaffinecomponent |  |
| naturalgradientperelementscalecomponent |  |
| noopcomponent | Not needed for inference. |
| normalizecomponent |  |
| parallelcomponent |  |
| pnormcomponent |  |
| rectifiedlinearcomponent |  |
| rescale |  |
| sigmoid |  |
| sigmoidcomponent |  |
| softmax |  |
| softmaxComponent |  |
| specaugmenttimemaskcomponent | Not needed for inference. |
| splicecomponent |  |
| tanhcomponent |  |
| tdnncomponent |  |
| timeheightconvolutioncomponent |  |


## ONNX Supported Operators

### Standard ONNX Operators

| ONNX Operator Name |
| :----------|
| Abs |
| Acos |
| Acosh |
| And |
| ArgMin | 
| ArgMax | 
| Asin |
| Asinh |
| Atan |
| ATen |
| Atanh |
| AveragePool |
| BatchNormalization |
| BitShift |
| Cast |
| CastLike |
| Ceil |
| Clip |
| Concat |
| Constant |
| ConstantOfShape |
| Conv |
| ConvInteger |
| ConvTranspose |
| Compress |
| Cos |
| Cosh |
| ConstantFill |
| CumSum |
| DepthToSpace |
| DequantizeLinear |
| Div |
| Dropout |
| Einsum |
| Elu |
| Equal |
| Erf |
| Exp |
| Expand |
| EyeLike |
| Flatten |
| Floor |
| Gather |
| GatherElements |
| GatherND |
| Gemm |
| GlobalAveragePool |
| GlobalLpPool |
| GlobalMaxPool |
| Greater |
| GRU |
| Hardmax |
| HardSigmoid |
| HardSwish |
| Identity |
| If |
| ImageScaler |
| InstanceNormalization |
| LeakyRelu |
| Less |
| Log |
| LogSoftmax |
| Loop |
| LpNormalization |
| LRN |
| LSTM |
| MatMulInteger |
| MatMul |
| MaxPool |
| Max |
| Mean |
| MeanVarianceNormalization |
| Min |
| Mod |
| Mul |
| Neg |
| NonMaxSuppression |
| NonZero |
| Not |
| Or |
| OneHot |
| Pad |
| Pow |
| PRelu |
| QLinearConv |
| QLinearMatMul |
| QuantizeLinear |
| Range |
| RandomNormal |
| RandomNormalLike |
| RandomUniform |
| RandomUniformLike |
| Reciprocal |
| ReduceLogSum |
| ReduceLogSumExp |
| ReduceL1 |
| ReduceL2 |
| ReduceMax |
| ReduceMean |
| ReduceMin |
| ReduceProd |
| ReduceSum |
| ReduceSumSquare |
| Relu |
| Reshape |
| Resize |
| ReverseSequence |
| RNN |
| RoiAlign |
| Round |
| ScatterElements |
| ScatterND |
| Selu |
| Shape |
| Shrink |
| Sigmoid |
| Sign |
| Sin |
| Sinh |
| Size |
| Slice |
| Softmax |
| Softplus |
| Softsign |
| SpaceToDepth |
| Split |
| Sqrt |
| Squeeze |
| Sub |
| Sum |
| Tan |
| Tanh |
| ThresholdedRelu |
| Tile |
| TopK |
| Transpose |
| Unsqueeze |
| Where |
| Xor |

### Deprecated ONNX Operators (Supported)

| ONNX Operator Name |
| :----------|
| Affine |
| Crop |
| Scatter |
| Upsample |

### Operators From the org.openvinotoolkit Domain

| Custom ONNX Operator Name |
| :----------|
| DeformableConv2D |
| DetectionOutput |
| ExperimentalDetectronDetectionOutput |
| ExperimentalDetectronGenerateProposalsSingleImage |
| ExperimentalDetectronGroupNorm |
| ExperimentalDetectronPriorGridGenerator |
| ExperimentalDetectronROIFeatureExtractor |
| ExperimentalDetectronTopKROIs |
| FakeQuantize |
| GroupNorm |
| Normalize |
| PriorBox |
| PriorBoxClustered |
| Swish |

### Operators From the com.microsoft Domain

| Custom ONNX Operator Name |
| :----------|
| Attention |
| BiasGelu |
| EmbedLayerNormalization |
| SkipLayerNormalization |

## PaddlePaddle Supported Operators

paddlepaddle>=2.1

| Operator Name in PaddlePaddle| Limitations|
| :----------| :----------|
| adpative_pool2d | The *`NHWC`* data_layout is not supported. |
| arg_max | The *`int32'`* output data_type is not supported. |
| assign_value |  |
| batch_norm |  |
| bilinear_interp | The *`NCW`*, *`NWC`*, *`NHWC`*, *`NCDHW`*, *`NDHWC`* data_layout are not supported. |
| bilinear_interp_v2 | The *`NCW`*, *`NWC`*, *`NHWC`*, *`NCDHW`*, *`NDHWC`* data_layout are not supported. |
| bmm |  |
| cast |  |
| clip |  |
| concat |  |
| conv2d | The *`NHWC`* data_layout is not supported. |
| depthwise_conv2d | The *`NHWC`* data_layout is not supported. |
| deformable_conv |  |
| elementwise_add |  |
| elementwise_div |  |
| elementwise_max |  |
| elementwise_min |  |
| elementwise_mul |  |
| elementwise_pow |  |
| elementwise_sub |  |
| equal |  |
| expand_v2 |  |
| exp |  |
| fill_any_like |  |
| fill_constant_batch_size_like |  |
| fill_constant |  |
| flatten_contiguous_range |  |
| gelu |  |
| greater_equal |  |
| hard_sigmoid |  |
| hard_swish |  |
| layer_norm |  |
| leaky_relu |  |
| log |  |
| logical_not |  |
| lookup_table_v2 |  |
| matmul |  |
| matmul_v2 |  |
| matrix_nms | Only supports IE CPU plugin with *"number of selected boxes"* static shape(e.g.: *`min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)`*). |
| max_pool2d_with_index |  |
| mul |  |
| multiclass_nms3 | Only supports IE CPU plugin with *"number of selected boxes"* static shape(e.g.: *`min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)`*). |
| nearest_interp | 'The *`NCW`*, *`NWC`*, *`NHWC`*, *`NCDHW`*, *`NDHWC`* data_layout are not supported. |
| nearest_interp_v2 | The *`NCW`*, *`NWC`*, *`NHWC`*, *`NCDHW`*, *`NDHWC`* data_layout are not supported. |
| pad3d | The *`Circular`* mode is not supported. |
| pow |  |
| pool2d | The *`NHWC`* data_layout is not supported. |
| prior_box |  |
| range |  |
| relu |  |
| relu6 |  |
| reshape2 |  |
| rnn | The *`SimpleRNN`* and *`GRU`* modes are not supported. |
| scale |  |
| shape |  |
| slice |  |
| softmax |  |
| sigmoid |  |
| split |  |
| squeeze2 |  |
| stack |  |
| tanh |  |
| transpose2 |  |
| unsqueeze2 |  |
| yolo_box |  |
