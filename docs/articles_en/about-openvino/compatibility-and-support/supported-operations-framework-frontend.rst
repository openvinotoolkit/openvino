.. {#openvino_resources_supported_operations_frontend}

Supported Operations - by Framework Frontend
============================================


.. meta::
   :description: Check the list of operations supported by OpenVINO Framework Frontend.


This page lists operations supported by OpenVINO Framework Frontend.


.. tab-set::

   .. tab-item:: PyTorch

      ==========================================  ==========================================================================================
       PyTorch Supported Operations                Limitations
      ==========================================  ==========================================================================================
      aten::__and__
      aten::__derive_index
      aten::__getitem__
      aten::__not__
      aten::__or__
      aten::__range_length
      aten::__xor__
      aten::_convolution
      aten::_convolution_mode
      aten::_native_multi_head_attention
      aten::_pack_padded_sequence
      aten::_pad_packed_sequence
      aten::_set_item
      aten::_shape_as_tensor
      aten::_unique2
      aten::_upsample_bicubic2d_aa
      aten::_upsample_bilinear2d_aa
      aten::_weight_norm
      aten::abs
      aten::abs_
      aten::acos
      aten::acos_
      aten::acosh
      aten::acosh_
      aten::adaptive_avg_pool1d
      aten::adaptive_avg_pool2d
      aten::adaptive_avg_pool3d
      aten::adaptive_max_pool1d
      aten::adaptive_max_pool2d
      aten::adaptive_max_pool3d
      aten::add
      aten::add_
      aten::addcmul
      aten::addmm
      aten::alias
      aten::alias_copy
      aten::all
      aten::amax
      aten::amin
      aten::append                                 Supported in limited set of patterns
      aten::arange
      aten::argmax
      aten::argmin
      aten::argsort
      aten::as_strided
      aten::as_tensor
      aten::asin
      aten::asin_
      aten::asinh
      aten::asinh_
      aten::atan
      aten::atan_
      aten::atanh
      aten::atanh_
      aten::avg_pool1d
      aten::avg_pool2d
      aten::avg_pool3d
      aten::baddbmm
      aten::batch_norm
      aten::bitwise_and
      aten::bitwise_not
      aten::bitwise_or
      aten::bitwise_xor
      aten::bmm
      aten::Bool
      aten::broadcast_tensors                      Supported in limited set of patterns
      aten::broadcast_to
      aten::bucketize
      aten::cat
      aten::cdist
      aten::ceil
      aten::ceil_
      aten::celu
      aten::celu_
      aten::channel_shuffle
      aten::chunk                                  Supported in limited set of patterns
      aten::clamp
      aten::clamp_
      aten::clamp_max
      aten::clamp_min
      aten::clip
      aten::clip_
      aten::clone
      aten::complex                                Supported in limited set of patterns
      aten::concat
      aten::contiguous
      aten::conv1d
      aten::conv2d
      aten::conv3d
      aten::conv_transpose1d
      aten::conv_transpose2d
      aten::conv_transpose3d
      aten::convolution
      aten::copy
      aten::copy_
      aten::cos
      aten::cos_
      aten::cosh
      aten::cosh_
      aten::cross
      aten::cumsum
      aten::dequantize
      aten::detach
      aten::dim
      aten::div
      aten::div_
      aten::dot
      aten::dropout
      aten::dropout_
      aten::einsum                                 Supported in limited set of patterns
      aten::elu
      aten::elu_
      aten::embedding
      aten::embedding_bag
      aten::empty
      aten::empty_like
      aten::eq
      aten::erf
      aten::erf_
      aten::erfc
      aten::erfc_
      aten::exp
      aten::exp_
      aten::expand
      aten::expand_as
      aten::expm1
      aten::expm1_
      aten::eye
      aten::fake_quantize_per_channel_affine
      aten::fake_quantize_per_tensor_affine
      aten::feature_dropout
      aten::fft_irfftn                             Supported in limited set of patterns
      aten::fft_rfftn                              Supported in limited set of patterns
      aten::fill
      aten::fill_
      aten::fill_diagonal_
      aten::flatten
      aten::flip
      aten::floor
      aten::floor_
      aten::floor_divide
      aten::floor_divide_
      aten::floordiv
      aten::fmod
      aten::frobenius_norm
      aten::full
      aten::full_like
      aten::gather
      aten::gcd
      aten::ge
      aten::gelu
      aten::glu
      aten::grid_sampler
      aten::group_norm
      aten::gru
      aten::gt
      aten::hardsigmoid
      aten::hardsigmoid_
      aten::hardswish
      aten::hardswish_
      aten::hardtanh
      aten::hardtanh_
      aten::im2col
      aten::imag                                   Supported in limited set of patterns
      aten::index                                  Supported in limited set of patterns
      aten::index_add
      aten::index_add_
      aten::index_copy_
      aten::index_put_
      aten::index_select
      aten::instance_norm
      aten::Int
      aten::IntImplicit
      aten::inverse
      aten::is_grad_enabled
      aten::is_nonzero
      aten::item
      aten::layer_norm
      aten::le
      aten::leaky_relu
      aten::leaky_relu_
      aten::len
      aten::lift
      aten::lift_fresh
      aten::lift_fresh_copy
      aten::linalg_cross
      aten::linalg_inv
      aten::linalg_matrix_norm
      aten::linalg_norm
      aten::linalg_vector_norm
      aten::linear
      aten::linspace
      aten::log
      aten::log10
      aten::log10_
      aten::log1p
      aten::log1p_
      aten::log2
      aten::log2_
      aten::log_
      aten::log_sigmoid
      aten::log_softmax
      aten::logical_and
      aten::logical_not
      aten::logical_or
      aten::logical_xor
      aten::lstm
      aten::lt
      aten::masked_fill
      aten::masked_fill_
      aten::masked_scatter
      aten::masked_scatter_
      aten::matmul
      aten::max
      aten::max_pool1d
      aten::max_pool1d_with_indices
      aten::max_pool2d
      aten::max_pool2d_with_indices
      aten::max_pool3d
      aten::max_pool3d_with_indices
      aten::maximum
      aten::mean
      aten::meshgrid
      aten::min
      aten::minimum
      aten::mish
      aten::mish_
      aten::mm
      aten::movedim
      aten::mul
      aten::mul_
      aten::multinomial
      aten::multiply
      aten::multiply_
      aten::mv
      aten::narrow
      aten::ne
      aten::neg
      aten::new_empty
      aten::new_full
      aten::new_ones
      aten::new_zeros
      aten::nonzero
      aten::nonzero_numpy                          Supported in limited set of patterns
      aten::norm
      aten::normal
      aten::normal_
      aten::numel
      aten::numpy_T
      aten::one_hot
      aten::ones
      aten::ones_like
      aten::outer
      aten::pad
      aten::pairwise_distance
      aten::permute
      aten::pixel_shuffle
      aten::pixel_unshuffle
      aten::pow
      aten::pow_
      aten::prelu
      aten::prod
      aten::quantize_per_channel
      aten::quantize_per_tensor
      aten::rand
      aten::rand_like
      aten::randint
      aten::randn
      aten::randn_like
      aten::real                                   Supported in limited set of patterns
      aten::reciprocal
      aten::reciprocal_
      aten::reflection_pad2d                       Supported in limited set of patterns
      aten::relu
      aten::relu6
      aten::relu6_
      aten::relu_
      aten::remainder
      aten::repeat
      aten::repeat_interleave
      aten::reshape
      aten::reshape_as
      aten::resolve_conj
      aten::resolve_neg
      aten::rnn_relu
      aten::rnn_tanh
      aten::roll
      aten::round
      aten::rsqrt
      aten::rsub
      aten::ScalarImplicit
      aten::scaled_dot_product_attention
      aten::scatter
      aten::scatter_
      aten::scatter_add
      aten::scatter_add_
      aten::scatter_reduce
      aten::scatter_reduce_
      aten::select
      aten::selu
      aten::selu_
      aten::sigmoid
      aten::sigmoid_
      aten::sign
      aten::silu
      aten::silu_
      aten::sin
      aten::sin_
      aten::sinh
      aten::sinh_
      aten::size
      aten::slice
      aten::softmax
      aten::softplus
      aten::sort
      aten::split                                  Supported in limited set of patterns
      aten::split_with_sizes                       Supported in limited set of patterns
      aten::sqrt
      aten::square
      aten::squeeze
      aten::stack                                  Supported in limited set of patterns
      aten::std
      aten::std_mean
      aten::sub
      aten::sub_
      aten::sum
      aten::swapaxes
      aten::t
      aten::t_
      aten::take_along_dim
      aten::tan
      aten::tan_
      aten::tanh
      aten::tanh_
      aten::tensor
      aten::tensor_split                           Supported in limited set of patterns
      aten::tile
      aten::to
      aten::topk
      aten::transpose
      aten::tril
      aten::tril_
      aten::triu
      aten::triu_
      aten::type_as
      aten::unbind                                 Supported in limited set of patterns
      aten::unflatten
      aten::unfold
      aten::unsqueeze
      aten::unsqueeze_
      aten::upsample_bicubic2d
      aten::upsample_bilinear2d
      aten::upsample_linear1d
      aten::upsample_nearest1d
      aten::upsample_nearest2d
      aten::upsample_nearest3d
      aten::upsample_trilinear3d
      aten::var
      aten::var_mean
      aten::view
      aten::view_as
      aten::where
      aten::zero_
      aten::zeros
      aten::zeros_like
      prim::Constant
      prim::device
      prim::DictConstruct                          Supported in limited set of patterns
      prim::GetAttr
      prim::If
      prim::is_cuda
      prim::ListConstruct
      prim::ListUnpack
      prim::Loop
      prim::max                                    Supported in limited set of patterns
      prim::min                                    Supported in limited set of patterns
      prim::NumToTensor
      prim::PythonOp
      prim::requires_grad
      prim::TupleConstruct                         Supported in limited set of patterns
      prim::TupleIndex
      prim::TupleUnpack                            Supported in limited set of patterns
      prim::type
      quantized::add
      quantized::add_relu
      quantized::cat
      quantized::conv2d
      quantized::conv2d_relu
      quantized::hardswish
      quantized::linear
      quantized::mul
      torchvision::deform_conv2d
      torchvision::nms
      torchvision::roi_align
      ==========================================  ==========================================================================================

   .. tab-item:: ONNX

      ==========================================  ==========================================================================================
       ONNX Supported Operations (standard)        Limitations
      ==========================================  ==========================================================================================
       Abs
       Acos
       Acosh
       Add
       And
       ArgMin
       ArgMax
       Asin
       Asinh
       Atan
       ATen
       Atanh
       AveragePool
       BatchNormalization
       BitShift
       Cast
       CastLike
       Ceil
       Clip
       Concat
       Constant
       ConstantOfShape
       Conv
       ConvInteger
       ConvTranspose
       Compress
       Cos
       Cosh
       ConstantFill
       CumSum
       DepthToSpace
       DequantizeLinear
       Div
       Dropout
       Einsum
       Elu
       Equal
       Erf
       Exp
       Expand
       EyeLike
       Flatten
       Floor
       Gather
       GatherElements
       GatherND
       Gemm
       GlobalAveragePool
       GlobalLpPool
       GlobalMaxPool
       Greater
       GRU
       Hardmax
       HardSigmoid
       HardSwish
       Identity
       If
       ImageScaler
       InstanceNormalization
       LeakyRelu
       Less
       Log
       LogSoftmax
       Loop
       LpNormalization
       LRN
       LSTM
       MatMulInteger
       MatMul
       MaxPool
       Max
       Mean
       MeanVarianceNormalization
       Min
       Mod
       Mul
       Neg
       NonMaxSuppression
       NonZero
       Not
       Or
       OneHot
       Pad
       Pow
       PRelu
       QLinearConv
       QLinearMatMul
       QuantizeLinear
       Range
       RandomNormal
       RandomNormalLike
       RandomUniform
       RandomUniformLike
       Reciprocal
       ReduceLogSum
       ReduceLogSumExp
       ReduceL1
       ReduceL2
       ReduceMax
       ReduceMean
       ReduceMin
       ReduceProd
       ReduceSum
       ReduceSumSquare
       Relu
       Reshape
       Resize
       ReverseSequence
       RNN
       RoiAlign
       Round
       ScatterElements
       ScatterND
       Selu
       Shape
       Shrink
       Sigmoid
       Sign
       Sin
       Sinh
       Size
       Slice
       Softmax
       Softplus
       Softsign
       SpaceToDepth
       Split
       Sqrt
       Squeeze
       Sub
       Sum
       Tan
       Tanh
       ThresholdedRelu
       Tile
       TopK
       Transpose
       Unsqueeze
       Where
       Xor
      ==========================================  ==========================================================================================

      ==========================================  ==========================================================================================
       ONNX Supported Operations (deprecated)      Limitations
      ==========================================  ==========================================================================================
       Affine
       Crop
       Scatter
       Upsample
      ==========================================  ==========================================================================================

      ======================================================================  ==============================================================
       ONNX Supported Operations (custom - the org.openvinotoolkit Domain)     Limitations
      ======================================================================  ==============================================================
       DeformableConv2D
       DetectionOutput
       ExperimentalDetectronDetectionOutput
       ExperimentalDetectronGenerateProposalsSingleImage
       ExperimentalDetectronGroupNorm
       ExperimentalDetectronPriorGridGenerator
       ExperimentalDetectronROIFeatureExtractor
       ExperimentalDetectronTopKROIs
       FakeQuantize
       GroupNorm
       Normalize
       PriorBox
       PriorBoxClustered
       Swish
      ======================================================================  ==============================================================

      ======================================================================  ==============================================================
       ONNX Supported Operations (custom - com.microsoft Domain)               Limitations
      ======================================================================  ==============================================================
       Attention
       BiasGelu
       EmbedLayerNormalization
       SkipLayerNormalization
      ======================================================================  ==============================================================

   .. tab-item:: PaddlePaddle

      ======================================================================  ==============================================================
       PaddlePaddle Supported Operations (v. >= 2.1)                           Limitations
      ======================================================================  ==============================================================
       arg_max                                                                 The ``int32`` output data_type is not supported.
       adaptive_pool2d                                                         The ``NHWC`` data_layout is not supported.
       assign
       assign_value
       batch_norm
       bicubic_interp
       bilinear_interp                                                         ``NCW``, ``NWC``, ``NHWC``, ``NCDHW``, ``NDHWC`` data_layout are not supported
       bmm
       box_coder
       cast
       ceil
       clip
       concat
       conditional_block
       conv2d                                                                  ``NHWC`` data_layout is not supported
       conv2d_transpose
       cumsum
       deformable_conv
       depthwise_conv2d                                                        ``NHWC`` data_layout is not supported.
       depthwise_conv2d_transpose
       dropout
       elementwise_add
       elementwise_div
       elementwise_floordiv
       elementwise_max
       elementwise_min
       elementwise_mod
       elementwise_mul
       elementwise_pow
       elementwise_sub
       equal
       exp
       expand
       fill_any_like
       fill_constant
       fill_constant_batch_size_like
       flatten_contiguous_range
       floor
       gather
       gather_nd
       gelu
       generate_proposals
       greater_equal
       greater_than
       group_norm
       hard_sigmoid
       hard_swish
       layer_norm
       leaky_relu
       less_than
       linear_interp
       log
       logical_and
       logical_not
       logical_or
       logical_xor
       lookup_table
       matmul
       matrix_nms                                                              Only supports CPU plugin with "number of selected boxes" static shape (e.g.: ``min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)``).
       max_pool2d_with_index
       meshgrid
       multiclass_nms                                                          Only supports CPU plugin with "number of selected boxes" static shape (e.g.: ``min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)``).
       nearest_interp                                                          ``NCW``, ``NWC``, ``NHWC``, ``NCDHW``, ``NDHWC`` data_layout are not supported.
       not_equal
       p_norm
       pad3d                                                                   ``Circular`` mode is not supported.
       pool2d                                                                  ``NHWC`` data_layout is not supported.
       pow
       prior_box
       range
       reduce_max
       reduce_mean
       reduce_min
       reduce_prod
       reduce_sum
       relu
       reshape
       reverse
       rnn                                                                     ``SimpleRNN`` and ``GRU`` modes are not supported.
       roi_align
       scale
       select_input
       shape
       sigmoid
       slice
       softmax
       softplus
       split
       sqrt
       squeeze
       stack
       strided_slice
       sum
       swish
       sync_batch_norm
       tanh
       tile
       top_k
       transpose
       trilinear_interp
       unsqueeze
       where
       where_index
       while
       yolo_box
      ======================================================================  ==============================================================

   .. tab-item:: TensorFlow

      ==========================================  ==========================================================================================
       TensorFlow Supported Operations             Limitations
      ==========================================  ==========================================================================================
       Abs
       Acosh
       Add
       AddV2
       AddN
       All
       Any
       ArgMax
       ArgMin
       Asinh
       Assert                                      Not needed for inference.
       Assign                                      Not needed for inference.
       AssignSub                                   Not needed for inference.
       Atanh
       AvgPool
       AvgPoolV2                                   Supported only for constant-foldable ``kernel_size`` and strides inputs.
       AvgPool3D
       BatchMatMul
       BatchMatMulV2
       BatchToSpaceND
       BiasAdd
       BlockLSTM
       Bucketize                                   CPU only.
       BroadcastTo
       Cast
       Ceil
       ClipByValue
       Concat
       ConcatV2
       Const
       Conv2D
       Conv2DBackpropInput
       Conv3D
       Conv3DBackpropInputV2
       Cos
       Cosh
       CropAndResize                               ``method`` = ``bilinear`` only.
       CTCGreedyDecoder                            Supported only with decoded indices output in a dense format.
       CTCLoss                                     Supported only with decoded indices input in a dense format.
       CumSum
       DepthToSpace
       DepthwiseConv2dNative
       Einsum                                      Supported only with equation that does not contain repeated labels within a subscript.
       Elu
       EmptyTensorList                             Supported only when it is part of a sub-graph of the special form.
       Enter                                       Supported only when it is fused to the TensorIterator layer.
       Equal
       Erf
       Exit                                        Supported only when it is fused to the TensorIterator layer.
       Exp
       ExpandDims
       ExperimentalSparseWeightedSum               CPU only.
       ExtractImagePatches
       EuclideanNorm
       FakeQuantWithMinMaxVars
       FakeQuantWithMinMaxVarsPerChannel
       FFT                                         Supported only when it is part of a sub-graph of the special form.
       FFT2D                                       Supported only when it is part of a sub-graph of the special form.
       FFT3D                                       Supported only when it is part of a sub-graph of the special form.
       FIFOQueueV2                                 Supported only when it is part of a sub-graph of the special form.
       Fill
       Floor
       FloorDiv
       FloorMod
       FusedBatchNorm
       FusedBatchNormV2
       FusedBatchNormV3
       Gather
       GatherNd
       GatherTree
       GatherV2
       Greater
       GreaterEqual
       Identity                                    Not needed for shape inference.
       IdentityN
       IFFT                                        Supported only when it is part of a sub-graph of the special form.
       IFFT2D                                      Supported only when it is part of a sub-graph of the special form.
       IFFT3D                                      Supported only when it is part of a sub-graph of the special form.
       IteratorGetNext                             Supported only when it is part of a sub-graph of the special form.
       LRN
       LeakyRelu
       Less
       LessEqual
       Log
       Log1p
       LogicalAnd
       LogicalOr
       LogicalNot
       LogSoftmax
       LookupTableInsertV2                         Supported only when it is part of a sub-graph of the special form.
       LoopCond                                    Supported only when it is fused to the TensorIterator layer.
       MatMul
       Max
       MaxPool
       MaxPoolV2                                   Supported only for constant-foldable ``kernel_size`` and strides inputs.
       MaxPool3D
       Maximum
       Mean
       Merge                                       Supported only when it is fused to the TensorIterator layer.
       Min
       Minimum
       MirrorPad
       Mod
       Mul
       Neg
       NextIteration                               Supported only when it is fused to the TensorIterator layer.
       NonMaxSuppressionV2
       NonMaxSuppressionV3
       NonMaxSuppressionV4
       NonMaxSuppressionV5
       NotEqual
       NoOp
       OneHot
       Pack
       Pad
       PadV2
       Placeholder
       PlaceholderWithDefault
       Prod
       QueueDequeue                                Supported only when it is part of a sub-graph of the special form.
       QueueDequeueUpToV2                          Supported only when it is part of a sub-graph of the special form.
       QueueDequeueV2                              Supported only when it is part of a sub-graph of the special form.
       RandomUniform
       RandomUniformInt
       Range
       Rank
       RealDiv
       Reciprocal
       Relu
       Relu6
       Reshape
       ResizeBilinear
       ResizeNearestNeighbor
       ResourceGather
       ReverseSequence
       ReverseV2                                   Supported only when it can be converted to the ReverseSequence operation.
       Roll
       Round
       Pow
       Rsqrt
       ScatterNd
       Select
       SelectV2
       Shape
       Sigmoid
       Sin
       Sinh
       Size
       Slice
       Softmax
       Softplus
       Softsign
       SpaceToBatchND
       SpaceToDepth
       SparseFillEmptyRows                         Supported only when it is part of a sub-graph of the special form.
       SparseReshape                               Supported only when it is part of a sub-graph of the special form.
       SparseSegmentSum                            Supported only when it is part of a sub-graph of the special form.
       SparseSegmentMean                           Supported only when it is part of a sub-graph of the special form.
       SparseToDense                               CPU only
       Split
       SplitV
       Sqrt
       Square
       SquaredDifference
       Square
       Squeeze                                     Cases in which squeeze axis is not specified are not supported.
       StatelessWhile
       StopGradient                                Not needed for shape inference.
       StridedSlice                                Supported only for constant-foldable ``begin``, ``end``, and ``strides`` inputs.
       Sub
       Sum
       Swish
       swish_f32
       Switch                                      Control flow propagation.
       Tan
       Tanh
       TensorArrayGatherV3                         Supported only when it is fused to the TensorIterator layer.
       TensorArrayReadV3                           Supported only when it is fused to the TensorIterator layer.
       TensorArrayScatterV3                        Supported only when it is fused to the TensorIterator layer.
       TensorArraySizeV3                           Supported only when it is fused to the TensorIterator layer.
       TensorArrayV3                               Supported only when it is fused to the TensorIterator layer.
       TensorArrayWriteV3                          Supported only when it is fused to the TensorIterator layer.
       TensorListPushBack                          Supported only when it is part of a sub-graph of the special form.
       Tile
       TopkV2
       Transpose
       Unpack
       Variable
       VariableV2
       Where                                       Supported only when it is part of a sub-graph of the special form.
       ZerosLike
      ==========================================  ==========================================================================================

   .. tab-item:: TensorFlow Lite

      ==========================================  ===============================================================================
      TensorFlow Lite Supported Operations         Limitations
      ==========================================  ===============================================================================
       ABS
       ADD
       ADD_N
       ARG_MAX
       ARG_MIN
       AVERAGE_POOL_2D
       BATCH_MATMUL
       BATCH_TO_SPACE_ND
       BROADCAST_ARGS
       BROADCAST_TO
       CAST
       CEIL
       COMPLEX_ABS                                 Supported in a specific pattern with RFFT2D
       CONCATENATION
       CONV_2D
       COS
       DEPTH_TO_SPACE
       DEPTHWISE_CONV_2D
       DEQUANTIZE
       DIV
       ELU
       EQUAL
       EXP
       EXPAND_DIMS
       FILL
       FLOOR
       FLOOR_DIV
       FLOOR_MOD
       FULLY_CONNECTED
       GATHER
       GATHER_ND
       GREATER
       GREATER_EQUAL
       HARD_SWISH
       L2_NORMALIZATION
       LEAKY_RELU
       LESS
       LESS_EQUAL
       LOG
       LOG_SOFTMAX
       LOGICAL_AND
       LOGICAL_NOT
       LOGICAL_OR
       LOGISTIC
       MATRIX_DIAG
       MAX_POOL_2D
       MAXIMUM
       MEAN
       MINIMUM
       MIRROR_PAD
       MUL
       NEG
       NOT_EQUAL
       ONE_HOT
       PACK
       PAD
       PADV2
       POW
       PRELU
       QUANTIZE
       RANGE
       RANK
       REDUCE_ALL
       REDUCE_ANY
       REDUCE_MAX
       REDUCE_MIN
       REDUCE_PROD
       RELU
       RELU6
       RESHAPE
       RESIZE_BILINEAR
       RESIZE_NEAREST_NEIGHBOR
       REVERSE_V2
       RFFT2D                                      Supported in a specific pattern with COMPLEX_ABS
       ROUND
       RSQRT
       SCATTER_ND
       SEGMENT_SUM
       SELECT
       SELECT_V2
       SHAPE
       SIGN
       SIN
       SLICE
       SOFTMAX
       SPACE_TO_BATCH_ND
       SPACE_TO_DEPTH
       SPLIT
       SPLIT_V
       SQRT
       SQUARE
       SQUARED_DIFFERENCE
       SQUEEZE
       STRIDED_SLICE
       SUB
       SUM
       TANH
       TILE
       TOPK_V2
       TRANSPOSE
       TRANSPOSE_CONV
       UNIQUE
       UNPACK
       WHERE
       ZEROS_LIKE
      ==========================================  ===============================================================================

   .. tab-item:: TensorFlow2 Keras

      ==========================================  ==========================================================================================
       TensorFlow 2 Keras Supported Operations     Limitations
      ==========================================  ==========================================================================================
       ActivityRegularization
       Add
       AdditiveAttention
       AlphaDropout
       Attention
       Average
       AveragePooling1D
       AveragePooling2D
       AveragePooling3D
       BatchNormalization
       Bidirectional
       Concatenate
       Conv1D
       Conv1DTranspose                             Not supported if ``dilation`` is not equal to 1.
       Conv2D
       Conv2DTranspose
       Conv3D
       Conv3DTranspose
       Cropping1D
       Cropping2D
       Cropping3D
       Dense
       DenseFeatures                               Not supported for categorical and crossed features.
       DepthwiseConv2D
       Dot
       Dropout
       ELU
       Embedding
       Flatten
       GRU
       GRUCell
       GaussianDropout
       GaussianNoise
       GlobalAveragePooling1D
       GlobalAveragePooling2D
       GlobalAveragePooling3D
       GlobalMaxPool1D
       GlobalMaxPool2D
       GlobalMaxPool3D
       LSTM
       LSTMCell
       Lambda
       LayerNormalization
       LeakyReLU
       LocallyConnected1D
       LocallyConnected2D
       MaxPool1D
       MaxPool2D
       MaxPool3D
       Maximum
       Minimum
       Multiply
       PReLU
       Permute
       RNN                                         Not supported for some custom cells.
       ReLU
       RepeatVector
       Reshape
       Roll
       SeparableConv1D
       SeparableConv2D
       SimpleRNN
       SimpleRNNCell
       Softmax
       SpatialDropout1D
       SpatialDropout2D
       SpatialDropout3D
       StackedRNNCells
       Subtract
       ThresholdedReLU
       TimeDistributed
       UpSampling1D
       UpSampling2D
       UpSampling3D
       ZeroPadding1D
       ZeroPadding2D
       ZeroPadding3D
      ==========================================  ==========================================================================================


