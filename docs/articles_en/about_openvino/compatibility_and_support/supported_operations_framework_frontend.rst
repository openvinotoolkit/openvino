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
      aten::__and__                                Only for boolean inputs
      aten::__getitem__                            Supported in limitted set of patterns
      aten::__not__ 
      aten::__or__ 
      aten::_convolution 
      aten::_convolution_mode 
      aten::_set_item 
      aten::abs 
      aten::acos 
      aten::acos_ 
      aten::acosh 
      aten::acosh_ 
      aten::adaptive_avg_pool2d 
      aten::adaptive_avg_pool3d 
      aten::adaptive_max_pool2d 
      aten::add 
      aten::add_ 
      aten::addcmul 
      aten::addmm 
      aten::append                                 Supported in limitted set of patterns
      aten::arange 
      aten::argsort 
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
      aten::bitwise_not 
      aten::bmm 
      aten::Bool 
      aten::cat 
      aten::ceil 
      aten::ceil_ 
      aten::chunk                                  Supported in limitted set of patterns
      aten::clamp 
      aten::clamp_max 
      aten::clamp_min 
      aten::clone 
      aten::contiguous 
      aten::conv_transpose1d 
      aten::conv_transpose2d 
      aten::conv_transpose3d 
      aten::conv1d 
      aten::conv2d 
      aten::conv3d 
      aten::convolution 
      aten::copy 
      aten::cos 
      aten::cos_ 
      aten::cosh 
      aten::cosh_ 
      aten::cumsum 
      aten::detach 
      aten::dim 
      aten::div 
      aten::div_ 
      aten::dropout 
      aten::dropout_ 
      aten::einsum                                 Supported in limitted set of patterns
      aten::elu 
      aten::embedding 
      aten::empty 
      aten::eq 
      aten::exp 
      aten::expand                                 Supported in limitted set of patterns
      aten::expand_as 
      aten::eye 
      aten::fill_ 
      aten::flatten 
      aten::floor 
      aten::floor_ 
      aten::floor_divide 
      aten::floordiv 
      aten::full 
      aten::full_like 
      aten::gather 
      aten::ge 
      aten::gelu 
      aten::glu 
      aten::grid_sampler 
      aten::group_norm 
      aten::gt 
      aten::hardsigmoid 
      aten::hardswish 
      aten::hardswish_ 
      aten::hardtanh 
      aten::hardtanh_ 
      aten::im2col 
      aten::index                                  Supported in limitted set of patterns
      aten::index_put_                             Supported in limitted set of patterns
      aten::index_select 
      aten::instance_norm 
      aten::Int 
      aten::IntImplicit 
      aten::is_grad_enabled 
      aten::layer_norm 
      aten::le 
      aten::leaky_relu 
      aten::leaky_relu_ 
      aten::len 
      aten::linalg_matrix_norm 
      aten::linalg_norm 
      aten::linalg_vector_norm 
      aten::linear 
      aten::log 
      aten::log_ 
      aten::log2 
      aten::log2_ 
      aten::lt 
      aten::masked_fill 
      aten::masked_fill_ 
      aten::matmul 
      aten::max 
      aten::max_pool1d 
      aten::max_pool2d 
      aten::max_pool3d 
      aten::mean 
      aten::meshgrid                               Supported in limitted set of patterns
      aten::min 
      aten::mm 
      aten::mul 
      aten::mul_ 
      aten::narrow 
      aten::ne 
      aten::neg 
      aten::new_empty 
      aten::new_full 
      aten::new_ones 
      aten::new_zeros 
      aten::nonzero 
      aten::nonzero_numpy                          Supported in limitted set of patterns
      aten::norm 
      aten::numel 
      aten::ones 
      aten::ones_like 
      aten::pad                                    Supported in limitted set of patterns
      aten::permute 
      aten::pow 
      aten::reciprocal 
      aten::relu 
      aten::relu_ 
      aten::relu6 
      aten::remainder 
      aten::repeat 
      aten::repeat_interleave 
      aten::reshape 
      aten::reshape_as 
      aten::roll 
      aten::rsqrt 
      aten::rsub 
      aten::ScalarImplicit 
      aten::scaled_dot_product_attention 
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
      aten::sort 
      aten::split                                  Supported in limitted set of patterns
      aten::split_with_sizes                       Supported in limitted set of patterns
      aten::sqrt 
      aten::square 
      aten::squeeze 
      aten::stack                                  Supported in limitted set of patterns
      aten::sub 
      aten::sum 
      aten::tan 
      aten::tan_ 
      aten::tanh 
      aten::tanh_ 
      aten::tensor 
      aten::to 
      aten::topk 
      aten::transpose 
      aten::tril 
      aten::triu 
      aten::type_as 
      aten::unbind                                 Supported in limitted set of patterns
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
      aten::where                                  Supported in limitted set of patterns
      aten::zeros 
      aten::zeros_like 
      prim::Constant                               String and None constant is only supported when used by supported operation
      prim::device 
      prim::DictConstruct                          Supported in limitted set of patterns
      prim::dtype                                  Supported in limitted set of patterns
      prim::GetAttr 
      prim::If 
      prim::is_cuda 
      prim::ListConstruct                          Supported in limitted set of patterns
      prim::ListUnpack                             Supported in limitted set of patterns
      prim::Loop 
      prim::max                                    Supported in limitted set of patterns
      prim::min                                    Supported in limitted set of patterns
      prim::NumToTensor 
      prim::PythonOp                               Supported only in tracing
      prim::requires_grad 
      prim::TupleConstruct                         Supported in limitted set of patterns
      prim::type
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
       matrix_nms                                                              Only supports IE CPU plugin with "number of selected boxes" static shape (e.g.: ``min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)``).
       max_pool2d_with_index 
       meshgrid 
       multiclass_nms                                                          Only supports IE CPU plugin with "number of selected boxes" static shape (e.g.: ``min(min(num_boxes, nms_top_k) * num_classes_output, keep_top_k)``).
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
 
   .. tab-item:: Caffe

      ==========================================  ==========================================================================================
       Caffe Supported Operations                  Limitations 
      ==========================================  ==========================================================================================
       Axpy
       BN
       BatchNorm 
       Bias 
       Binarization (Intel experimental) 
       Concat 
       Convolution 
       ConvolutionBinary 
       Crop 
       Deconvolution 
       DetectionOutput 
       Dropout                                     Not needed for inference.
       Eltwise 
       Flatten 
       GlobalInput 
       InnerProduct 
       Input 
       LRN 
       Normalize 
       Python                                      Supported only for the Python Proposal operation.
       Permute 
       Pooling 
       Power 
       PReLU 
       PriorBox 
       PriorBoxClustered 
       Proposal 
       PSROIPooling 
       ROIPooling 
       RegionYolo 
       ReorgYolo 
       ReLU 
       Resample 
       Reshape 
       Scale 
       ShuffleChannel 
       Sigmoid 
       Slice 
       Softmax 
       Tile 
      ==========================================  ==========================================================================================
            
   .. tab-item:: Kaldi

      ==========================================  ==========================================================================================
       Kaldi Supported Operations                  Limitations
      ==========================================  ==========================================================================================
       addshift 
       affinecomponent 
       affinecomponentpreconditionedonline 
       affinetransform 
       backproptruncationcomponent 
       batchnormcomponent 
       clipgradientcomponent                       Not needed for inference. 
       concat 
       convolutional1dcomponent 
       convolutionalcomponent 
       copy 
       dropoutmaskcomponent 
       elementwiseproductcomponent 
       fixedaffinecomponent 
       fixedbiascomponent 
       fixedscalecomponent 
       generaldropoutcomponent                     Not needed for inference. 
       linearcomponent 
       logsoftmaxcomponent 
       lstmnonlinearitycomponent 
       lstmprojected 
       lstmprojectedstreams 
       maxpoolingcomponent 
       naturalgradientaffinecomponent 
       naturalgradientperelementscalecomponent 
       noopcomponent                               Not needed for inference. 
       normalizecomponent 
       parallelcomponent 
       pnormcomponent 
       rectifiedlinearcomponent 
       rescale 
       sigmoid 
       sigmoidcomponent 
       softmax 
       softmaxComponent 
       specaugmenttimemaskcomponent                Not needed for inference. 
       splicecomponent 
       tanhcomponent 
       tdnncomponent 
       timeheightconvolutioncomponent 
      ==========================================  ==========================================================================================
            
   .. tab-item:: Apache MXNet

      ==========================================  ==========================================================================================
       Apache MXNet Supported Operations           Limitations
      ==========================================  ==========================================================================================
       _Plus 
       _contrib_arange_like 
       _contrib_box_nms 
       _contrib_DeformableConvolution 
       _contrib_DeformablePSROIPooling 
       _contrib_div_sqrt_dim 
       _contrib_MultiBoxDetection                  ``force_suppress`` = 1 is not supported, non-default variances are not supported. 
       _contrib_MultiBoxPrior 
       _contrib_Proposal 
       _copy                                       Not needed for inference 
       _div_scalar 
       _greater_scalar 
       _minus_scalar 
       _mul_scalar 
       _plus_scalar 
       _random_uniform                             Operation provides sequence from uniform distribution, but exact values won't match. 
       _rnn_param_concat 
       _arange 
       _contrib_AdaptiveAvgPooling2D               Converted to the Average Pooling with fixed paddings. 
       _maximum 
       _minimum 
       _np_roll 
       _zeros 
       add_n 
       arccosh 
       arcsinh 
       arctanh 
       batch_dot 
       broadcast_add 
       broadcast_div 
       broadcast_mul 
       broadcast_sub 
       BlockGrad 
       cumsum 
       div_scalar 
       elementwise_sub 
       elemwise_add 
       elemwise_mul 
       elemwise_sub 
       exp 
       expand_dims 
       greater_scalar 
       max 
       minus_scalar 
       null                                        Not needed for inference. 
       LayerNorm                                   ``output_mean_var`` = True is not supported. 
       repeat 
       rnn 
       rnn_param_concat 
       round 
       sigmoid 
       slice 
       SliceChannel 
       slice_axis 
       slice_channel 
       slice_like 
       softmax 
       stack 
       swapaxis 
       tile 
       transpose 
       zeros 
       Activation                                  Supported ``act_type`` = ``relu``, ``sigmoid``, ``softrelu`` or ``tanh``
       BatchNorm 
       Concat 
       Convolution 
       Crop                                        ``center_crop`` = 1 is not supported. 
       Custom                                      See :doc:`Custom Layers in Model Optimizer <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer>` 
       Deconvolution 
       DeformableConvolution 
       DeformablePSROIPooling 
       Dropout                                     Not needed for inference. 
       ElementWiseSum 
       Embedding 
       Flatten 
       FullyConnected 
       InstanceNorm 
       L2Normalization                             Only 4D input is supported. 
       LRN 
       LeakyReLU                                   Supported ``act_type`` = ``prelu``, ``elu``, ``leaky``, ``gelu``
       ones_like 
       Pad 
       Pooling 
       ROIPooling 
       ReLU 
       Reshape 
       ScaleShift 
       SoftmaxActivation 
       SoftmaxOutput 
       SoftSign 
       Take                                        The attribute ``mode`` is not supported.
       Tile 
       UpSampling 
       Where 
       zeros_like 
      ==========================================  ==========================================================================================
            
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
      

