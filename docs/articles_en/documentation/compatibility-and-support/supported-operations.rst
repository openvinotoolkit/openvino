Supported Operations
===============================================================================================

.. meta::
   :description: Check the operations supported by OpenVINO.


Here, you will find comprehensive information on operations supported by OpenVINO. The
conformance reports provide operation coverage for inference devices, while the tables list
operations available for all OpenVINO framework frontends.

Data as of OpenVINO 2024.4, 18 Oct. 2024.

**Device-operation conformance reports:**

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. button-link:: ../../_static/conformance_files/conformance_reports/opset_report_omz_static.html
         :color: primary
         :outline:
         :expand:

         ops with static shapes only

   .. grid-item::

      .. button-link:: ../../_static/conformance_files/conformance_reports/opset_report_omz_dynamic.html
         :color: primary
         :outline:
         :expand:

         ops including dynamic inputs


**Operations supported by OpenVINO frontend Frameworks:**

.. tab-set::

   .. tab-item:: PyTorch

      .. data-table::
         :class: modeldata stripe
         :name: TensorFlow_ops_v1
         :header-rows: 1
         :file:  ../../_static/conformance_files/pytorch_ops.csv
         :data-column-hidden: []
         :data-order: [[ 0, "asc" ]]
         :data-page-length: 10

   .. tab-item:: TensorFlow

      .. data-table::
         :class: modeldata stripe
         :name: TensorFlow_ops_v2
         :header-rows: 1
         :file:  ../../_static/conformance_files/tensorflow_ops.csv
         :data-column-hidden: []
         :data-order: [[ 0, "asc" ]]
         :data-page-length: 10

   .. tab-item:: PaddlePaddle

      .. data-table::
         :class: modeldata stripe
         :name: Paddle_ops
         :header-rows: 1
         :file:  ../../_static/conformance_files/paddlepaddle_ops.csv
         :data-column-hidden: []
         :data-order: [[ 0, "asc" ]]
         :data-page-length: 10

   .. tab-item:: ONNX

      ==========================================  ==========================================================================================
       ONNX Supported Operations (standard)        Limitations
      ==========================================  ==========================================================================================
       Abs
       Acos
       Acosh
       AdaptiveAvgPool2d
       Add
       Affine
       And
       ArgMax
       ArgMin
       Asin
       Asinh
       Atan
       Atanh
       ATen
       AveragePool
       BatchNormalization
       BitShift
       BitwiseAnd
       BitwiseNot
       BitwiseOr
       BitwiseXor
       BlackmanWindow
       Cast
       CastLike
       Ceil
       Celu
       Clip
       Compress
       Concat
       Constant
       ConstantFill
       ConstantOfShape
       Conv
       ConvInteger
       ConvTranspose
       Cos
       Cosh
       Crop
       CumSum
       DepthToSpace
       DequantizeLinear
       DFT
       Div
       Dropout
       DynamicQuantizeLinear
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
       Gelu
       Gemm
       GlobalAveragePool
       GlobalLpPool
       GlobalMaxPool
       Greater
       GreaterOrEqual
       GridSample
       GroupNormalization
       GRU
       HammingWindow
       HardSigmoid
       HardSwish
       Hardmax
       Identity
       If
       ImageScaler
       InstanceNormalization
       IsFinite
       IsInf
       IsNaN
       LayerNormalization
       LeakyRelu
       Less
       LessOrEqual
       Log
       LogSoftmax
       Loop
       LpNormalization
       LRN
       LSTM
       MatMul
       MatMulInteger
       Max
       MaxPool
       MaxRoiPool
       Mean
       MeanVarianceNormalization
       Min
       Mish
       MMCVRoIAlignRotated
       Mod
       Mul
       Multinomial
       Neg
       NMSRotated
       NonMaxSuppression
       NonZero
       Not
       OneHot
       Or
       Pad
       Pow
       PRelu
       QLinearConv
       QLinearMatMul
       QuantizeLinear
       RandomNormal
       RandomNormalLike
       RandomUniform
       RandomUniformLike
       Range
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
       Scan
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
       STFT
       Sub
       Sum
       Tan
       Tanh
       ThresholdedRelu
       Tile
       TopK
       Transpose
       Trilu
       Unique
       Unsqueeze
       Upsample
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
       Bias_Add
       BiasGelu
       Dynamic_Quantize_MatMul
       EmbedLayerNormalization
       Fast_Gelu
       Fused_Conv
       FusedGemm
       FusedMatMul
       MatMulIntegerToFloat
       MatMulNBits
       Pad
       QLinearActivation
       QLinearAdd
       QLinearMul
       QuickGelu
       Range
       SimplifiedLayerNormalization
       SkipLayerNormalization
       SkipSimplifiedLayerNormalization
      ======================================================================  ==============================================================


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


