# Supported Framework Layers {#openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers}

## Caffe\* Supported Layers

Standard Caffe\* layers:

| Layer Name in Caffe\* | Limitations |
|:---------- | :----------|
| Axpy | No |
| BN | No |
| BatchNorm | No |
| Bias | No |
| Concat | No |
| Convolution | No |
| Deconvolution | No |
| DetectionOutput | No |
| Dropout | Not needed for inference |
| Eltwise | No |
| Flatten | No |
| GlobalInput | No |
| InnerProduct | No |
| Input | No |
| LRN | No |
| Permute | No |
| Pooling | No |
| Power | No |
| ROIPooling | No |
| ReLU | No |
| Reshape | No |
| Scale | No |
| ShuffleChannel | No |
| Slice | No |
| Softmax | No |
| Tile | No |


## MXNet\* Supported Symbols

Standard MXNet\* symbols:

| Symbol Name in MXNet\*| Limitations|
| :----------| :----------|
| _Plus | No |
| _contrib_MultiBoxDetection | "force_suppress" = 1 is not supported, non-default variances are not supported |
| _contrib_MultiBoxPrior | No |
| _contrib_Proposal | No |
| _copy | Not needed for inference |
| _minus_scalar | No |
| _mul_scalar | No |
| _arange | No |
| _contrib_AdaptiveAvgPooling2D | Converted to the Average Pooling with fixed paddings |
| _maximum | No |
| _minimum | No |
| _np_roll | No |
| add_n | No |
| arccosh | No |
| arcsinh | No |
| arctanh | No |
| broadcast_add | No |
| broadcast_mul | No |
| cumsum | No |
| div_scalar | No |
| elementwise_sub | No |
| elemwise_add | No |
| elemwise_mul | No |
| exp | No |
| expand_dims | No |
| greater_scalar | No |
| minus_scalar | No |
| null | Not needed for inference |
| repeat | No |
| rnn | No |
| rnn_param_concat | No |
| round | No |
| sigmoid | No |
| slice | No |
| slice_axis | No |
| slice_channel | No |
| slice_like | No |
| stack | No |
| swapaxis | No |
| tile | No |
| transpose | No |
| zeros | No |
| Activation | supported "act_type" = "relu", "sigmoid", "softrelu" or "tanh" |
| BatchNorm | No |
| Concat | No |
| Convolution | No |
| Crop | "center_crop" = 1 is not supported |
| Custom | [Custom Layers in the Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md) |
| Deconvolution | No |
| DeformableConvolution | No |
| DeformablePSROIPooling | No |
| Dropout | Not needed for inference |
| ElementWiseSum | No |
| Embedding | No |
| Flatten | No |
| FullyConnected | No |
| InstanceNorm | No |
| L2Normalization | only 4D input is supported |
| LRN | No |
| LeakyReLU | supported "act_type" = "prelu", "elu", "leaky", "gelu" |
| Pad | No |
| Pooling | No |
| ROIPooling | No |
| ReLU | No |
| Reshape | No |
| ScaleShift | No |
| SoftmaxActivation | No |
| SoftmaxOutput | No |
| SoftSign | No |
| Take | The attribute 'mode' is not supported |
| Tile | No |
| UpSampling | No |
| Where | No |


## TensorFlow\* Supported Operations

Some TensorFlow\* operations do not match to any Inference Engine layer, but are still supported by the Model Optimizer and can be used on constant propagation path. These layers are labeled 'Constant propagation' in the table.

Standard TensorFlow\* operations:

| Operation Name in TensorFlow\* | Limitations|
| :----------| :----------|
| Acosh | No |
| Add | No |
| AddV2 | No |
| AddN | No |
| ArgMax | No |
| ArgMin | No |
| Asinh | No |
| Atanh | No |
| AvgPool | No |
| AvgPoolV2 | Supported only for constant-foldable kernel_size and strides inputs |
| BatchToSpaceND | No |
| BiasAdd | No |
| Bucketize | CPU only |
| BroadcastTo | No |
| Cast | No |
| Ceil | No |
| Concat | No |
| ConcatV2 | No |
| Const | No |
| Conv2D | No |
| Conv2DBackpropInput | No |
| Cos | No |
| Cosh | No |
| CropAndResize | "method" = "bilinear" only |
| CumSum | No |
| DepthToSpace| No |
| DepthwiseConv2dNative| No |
| Enter | Supported only when it is fused to the TensorIterator layer |
| Equal | No |
| Exit | Supported only when it is fused to the TensorIterator layer |
| Exp | No |
| ExpandDims | No |
| ExperimentalSparseWeightedSum | CPU only |
| ExtractImagePatches | No |
| EuclideanNorm | No |
| FakeQuantWithMinMaxVars | No |
| FakeQuantWithMinMaxVarsPerChannel | No |
| FFT | Supported only when it is part of a sub-graph of the special form |
| FFT2D | Supported only when it is part of a sub-graph of the special form |
| FFT3D | Supported only when it is part of a sub-graph of the special form |
| Fill | No |
| Floor | No |
| FloorDiv | No |
| FusedBatchNorm | No |
| FusedBatchNormV2 | No |
| FusedBatchNormV3 | No |
| Gather | No |
| GatherNd | No |
| GatherV2 | No |
| Greater | No |
| GreaterEqual | No |
| Identity | Not needed for shape inference |
| IFFT | Supported only when it is part of a sub-graph of the special form |
| IFFT2D | Supported only when it is part of a sub-graph of the special form |
| IFFT3D | Supported only when it is part of a sub-graph of the special form |
| LRN | No |
| Less | No |
| Log | No |
| Log1p | No |
| LogicalAnd | No |
| LogicalOr | No |
| LogicalNot | No |
| LogSoftmax | No |
| LoopCond | Supported only when it is fused to the TensorIterator layer |
| MatMul | No |
| Max | No |
| MaxPool | No |
| MaxPoolV2 | Supported only for constant-foldable kernel_size and strides inputs |
| Maximum | No |
| Mean | No |
| Merge | Supported only when it is fused to the TensorIterator layer |
| Min | No |
| Minimum | No |
| MirrorPad | No |
| Mul | No |
| Neg | No |
| NextIteration | Supported only when it is fused to the TensorIterator layer |
| NonMaxSuppressionV3 | No |
| NonMaxSuppressionV4 | No |
| NonMaxSuppressionV5 | No |
| NoOp | No |
| OneHot | No |
| Pack | No |
| Pad | No |
| PadV2 | No |
| Placeholder | No |
| PlaceholderWithDefault | No |
| Prod | No |
| Range | No |
| Rank | No |
| RealDiv | No |
| Relu | No |
| Relu6 | No |
| Reshape | No |
| ResizeBilinear | No |
| ResizeNearestNeighbor | No |
| ResourceGather| No |
| ReverseSequence | No |
| Roll | No |
| Round | No |
| Rsqrt | No |
| Shape | No |
| Sigmoid | No |
| Sin | No |
| Sinh | No |
| Size | No |
| Slice | No |
| Softmax | No |
| Softplus | No |
| Softsign | No |
| SpaceToBatchND | No |
| SparseToDense | CPU only |
| Split | No |
| SplitV | No |
| Sqrt | No |
| Square | No |
| SquaredDifference | No |
| Square| No |
| Squeeze | The case when squeeze axis is not specified is not supported |
| StopGradient | Not needed for shape inference |
| StridedSlice | Supported only for constant-foldable begin, end, and strides inputs |
| Sub | No |
| Sum | No |
| Swish | No |
| Switch | Control flow propagation |
| Tan | No |
| Tanh | No |
| TensorArrayGatherV3 | Supported only when it is fused to the TensorIterator layer |
| TensorArrayReadV3 | Supported only when it is fused to the TensorIterator layer |
| TensorArrayScatterV3 | Supported only when it is fused to the TensorIterator layer |
| TensorArraySizeV3 | Supported only when it is fused to the TensorIterator layer |
| TensorArrayV3 | Supported only when it is fused to the TensorIterator layer |
| TensorArrayWriteV3 | Supported only when it is fused to the TensorIterator layer |
| Tile | No |
| TopkV2 | No |
| Transpose | No |
| Unpack | No |
| Where | No |
| ZerosLike | No |


## TensorFlow 2 Keras\* Supported Operations

Standard TensorFlow 2 Keras\* operations:

| Operation Name in TensorFlow 2 Keras\* | Limitations|
| :----------| :----------|
| ActivityRegularization | No |
| Add | No |
| AdditiveAttention | No |
| AlphaDropout | No |
| Attention | No |
| Average | No |
| AveragePooling1D | No |
| AveragePooling2D | No |
| AveragePooling3D | No |
| BatchNormalization | No |
| Bidirectional | No |
| Concatenate | No |
| Conv1D | No |
| Conv1DTranspose | Not supported if dilation is not equal to 1 |
| Conv2D | No |
| Conv2DTranspose | No |
| Conv3D | No |
| Conv3DTranspose | No |
| Cropping1D | No |
| Cropping2D | No |
| Cropping3D | No |
| Dense | No |
| DenseFeatures | Not supported for categorical and crossed features |
| DepthwiseConv2D | No |
| Dot | No |
| Dropout | No |
| ELU | No |
| Embedding | No |
| Flatten | No |
| GRU | No |
| GRUCell | No |
| GaussianDropout | No |
| GaussianNoise | No |
| GlobalAveragePooling1D | No |
| GlobalAveragePooling2D | No |
| GlobalAveragePooling3D | No |
| GlobalMaxPool1D | No |
| GlobalMaxPool2D | No |
| GlobalMaxPool3D | No |
| LSTM | No |
| LSTMCell | No |
| Lambda | No |
| LayerNormalization | No |
| LeakyReLU | No |
| LocallyConnected1D | No |
| LocallyConnected2D | No |
| MaxPool1D | No |
| MaxPool2D | No |
| MaxPool3D | No |
| Maximum | No |
| Minimum | No |
| Multiply | No |
| PReLU | No |
| Permute | No |
| RNN | Not supported for some custom cells |
| ReLU | No |
| RepeatVector | No |
| Reshape | No |
| Roll | No |
| SeparableConv1D | No |
| SeparableConv2D | No |
| SimpleRNN | No |
| SimpleRNNCell | No |
| Softmax | No |
| SpatialDropout1D | No |
| SpatialDropout2D | No |
| SpatialDropout3D | No |
| StackedRNNCells | No |
| Subtract | No |
| ThresholdedReLU | No |
| TimeDistributed | No |
| UpSampling1D | No |
| UpSampling2D | No |
| UpSampling3D | No |
| ZeroPadding1D | No |
| ZeroPadding2D | No |
| ZeroPadding3D | No |

## Kaldi\* Supported Layers

Standard Kaldi\* Layers:

| Symbol Name in Kaldi\*| Limitations|
| :----------| :----------|
| addshift | No |
| affinecomponent | No |
| affinetransform | No |
| clipgradientcomponent | Not needed for inference |
| concat | No |
| convolutional1dcomponent | No |
| convolutionalcomponent | No |
| copy | No |
| Crop | No |
| elementwiseproductcomponent | No |
| fixedaffinecomponent | No |
| fixedbiascomponent | No |
| fixedscalecomponent | No |
| generaldropoutcomponent| Not needed for inference |
| linearcomponent | No |
| logsoftmaxcomponent | No |
| lstmnonlinearitycomponent | No |
| lstmprojected | No |
| lstmprojectedstreams | No |
| maxpoolingcomponent | No |
| naturalgradientaffinecomponent | No |
| naturalgradientperelementscalecomponent | No |
| noopcomponent | Not needed for inference |
| normalizecomponent | No |
| parallelcomponent | No |
| pnormcomponent | No |
| rectifiedlinearcomponent | No |
| rescale | No |
| sigmoid | No |
| softmax | No |
| softmaxComponent | No |
| softsign | No |
| specaugmenttimemaskcomponent | Not needed for inference |
| splicecomponent | No |
| tanhcomponent | No |
| tdnncomponent | No |
| timeheightconvolutioncomponent | No |


## ONNX\* Supported Operators

Standard ONNX\* operators:

| Symbol Name in ONNX\*| Limitations|
| :----------| :----------|
| Abs | No |
| Acos | No |
| Acosh | No |
| Add | No |
| Affine | No |
| ArgMax | No |
| ArgMin | No |
| Asin | No |
| Asinh | No |
| Atan | No |
| Atanh | No |
| AveragePool | No |
| BatchMatMul | No |
| BatchNormalization | No |
| Cast | No |
| Ceil | No |
| Clip | No |
| Concat | No |
| Constant | No |
| ConstantFill | No |
| ConstantOfShape | No |
| Conv | No |
| ConvTranspose |  |
| Cos | No |
| Cosh | No |
| Crop | No |
| CumSum | No |
| DequantizeLinear | No |
| DetectionOutput (Intel experimental) | No |
| Div | No |
| Dropout | Not needed for inference |
| Elu | No |
| Equal | No |
| Erf | No |
| Expand | No |
| FakeQuantize (Intel experimental) | No |
| Fill | No |
| Flatten | No |
| Floor | No |
| GRU | No |
| Gather | No |
| GatherElements | Doesn't work with negative indices |
| GatherND | No |
| GatherTree | No |
| Gemm | No |
| GlobalAveragePool | No |
| GlobalMaxPool | No |
| Greater | No |
| GreaterEqual | No |
| HardSigmoid | No |
| Identity | Not needed for inference |
| ImageScaler | No |
| LRN | No |
| LSTM | Peepholes are not supported |
| LeakyRelu | No |
| Less | No |
| LessEqual | No |
| Log | No |
| LogicalAnd | No |
| LogicalOr | No |
| LogSoftmax | No |
| Loop | No |
| MatMul | No |
| MaxPool | No |
| MeanVarianceNormalization | Reduction over the batch dimension is not supported, reduction over all dimensions except batch and channel ones is obligatory |
| Min | No |
| Mul | No |
| Neg | No |
| NonMaxSuppression | No |
| NonZero | No |
| Not | No |
| NotEqual | No |
| OneHot | No |
| Pad | No |
| Pow | No |
| PriorBox (Intel experimental) | No |
| QuantizeLinear | No |
| RNN | No |
| ROIAlign | No |
| Range | No |
| Reciprocal | No |
| ReduceL1 | No |
| ReduceL2 | No |
| ReduceMax | No |
| ReduceMean | No |
| ReduceMin | No |
| ReduceProd | No |
| ReduceSum | No |
| Relu | No |
| Reshape | No |
| Resize | Coordinate transformation mode `tf_crop_and_resize` is not supported, `nearest` mode is not supported for 5D+ inputs. |
| ReverseSequence | No |
| Round | No |
| Scatter | Supported if fuse-able to ScatterUpdate. MYRIAD only |
| ScatterND | No |
| ScatterElements | Supported if fuse-able to ScatterUpdate. MYRIAD only |
| Select | No |
| Shape | No |
| Sigmoid | No |
| Sign | No |
| Sin | No |
| Size | No |
| Slice | No |
| Softmax | No |
| Softplus | No |
| Softsign | No |
| SpaceToDepth | No |
| Sqrt | No |
| Squeeze | The case when squeeze axis is not specified is not supported |
| Sub | No |
| Sum | No |
| Tan | No |
| Tanh | No |
| ThresholdedRelu | No |
| TopK | No |
| Transpose | No |
| Unsqueeze | No |
| Upsample | No |
| Where | No |
| Xor | No |


## PaddlePaddle\* Supported Operators

Standard PaddlePaddle(paddlepaddle>=2.1)\* Operators:

| Operator Name in PaddlePaddle\*| Limitations|
| :----------| :----------|
| adpative_pool2d | Only supports the NCHW data_layout |
| arg_max | Only supports the 'Int64' output data_type |
| assign_value | No |
| batch_norm | No |
| bilinear_interp | Only supports the NCHW data_layout |
| bilinear_interp_v2 | Only supports the NCHW data_layout |
| bmm | No |
| cast | No |
| clip | No |
| concat | No |
| conv2d | Only supports the NCHW data_layout |
| depthwise_conv2d | Only supports the NCHW data_layout |
| deformable_conv | No |
| elementwise_add | No |
| elementwise_div | No |
| elementwise_max | No |
| elementwise_min | No |
| elementwise_mul | No |
| elementwise_pow | No |
| elementwise_sub | No |
| equal | No |
| expand_v2 | No |
| fill_constant_batch_size_like | No |
| fill_constant | No |
| flatten_contiguous_range | No |
| greater_equal | No |
| hard_sigmoid | No |
| hard_swish | No |
| leaky_relu | No |
| log | No |
| logical_not | No |
| matmul | No |
| matrix_nms | Only supports IE CPU plugin with 'number of selected boxes' static shape(eg: min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)) |
| max_pool2d_with_index | No |
| mul | No |
| multiclass_nms | Only supports IE CPU plugin with 'number of selected boxes' static shape(eg: min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)) |
| nearest_interp | Only supports the NCHW data_layout |
| nearest_interp_v2 | Only supports the NCHW data_layout |
| pad3d | Not supports the 'Circular' mode |
| pow | No |
| pool2d | Only supports the NCHW data_layout |
| range | No |
| relu | No |
| relu6 | No |
| reshape2 | No |
| rnn | Only supports the 'LSTM' Mode |
| scale | No |
| shape | No |
| slice | No |
| softmax | No |
| sigmoid | No |
| split | No |
| squeeze2 | No |
| transpose2 | No |
| unsqueeze2 | No |
| yolo_box | No |
