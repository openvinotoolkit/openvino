#ONNX layer test coverage

*Note: coverage field should be from 1 to 10 (depending) on operation test coverage or "-" if test do not exists*

| Operation            | MO Status | IE Status | Priority | Coverage |  
|:---------------------|:---------:|:---------:|:--------:|:--------:|
|Abs                   |           |           |    P2    |          |
|Acos                  |           |           |    ?     |          |
|Acosh                 |           |           |    ?     |          |
|Add                   |     +     |     +     |    P1    |    5     |
|Add-6                 |           |           |    P2    |          |
|And                   |           |           |    P3    |          |
|ArgMax                |           |           |    P3    |          |
|Asin                  |           |           |    ?     |          |
|ArgMin                |           |           |    P4    |          |
|Asinh                 |           |           |    ?     |          |
|Atan                  |           |           |    ?     |          |
|Atanh                 |           |           |    ?     |          |
|AveragePool           |     +     |     +     |    P1    |    5     |
|BatchNormalization    |     +     |    +      |    P1    |     5    |
|Cast                  |           |           |    ?     |          |
|Ceil                  |           |           |    ?     |          |
|Clip                  |           |           |    P3    |          |
|Compress              |           |           |    ?     |          |
|Concat                |     +     |    +      |    P1    |    10    |
|Constant              |           |           |    P1    |          |
|ConstantLike          |           |           |    ?     |          |
|Conv                  |           |           |    P1    |          |
|ConvTranspose         |           |           |    P1    |          |
|Cos                   |           |           |    ?     |          |
|Cosh                  |           |           |    ?     |          |
|DepthToSpace          |           |           |    P3    |          |
|Div                   |     +     |     +     |    P1    |    5     |
|Div-6                 |           |           |    P2    |          |
|Dropout               |           |           |    P3    |          |
|Elu                   |     +     |    +      |    P2    |    10    |
|Equal                 |           |           |    ?     |          |
|Exp                   |           |           |    P3    |          |
|Expand                |           |           |    ?     |          |
|EyeLike               |           |           |    ?     |          |
|Flatten               |    +      |    +      |    P1    |    10    |
|Floor                 |           |           |    ?     |          |
|GRU                   |           |           |    ?     |          |
|Gather                |           |           |    P2    |          |
|Gemm                  |           |           |    P1    |          |
|GlobalAveragePool     |           |           |    P1    |          |
|GlobalLpPool          |           |           |    P3    |          |
|GlobalMaxPool         |           |           |    P3    |          |
|Greater               |           |           |    ?     |          |
|HardSigmoid           |           |           |    ?     |          |
|Hardmax               |           |           |    ?     |          |
|Identity              |           |           |    ?     |          |
|If                    |           |           |    ?     |          |
|InstanceNormalization |           |           |    P2    |          |
|LRN                   |           |           |    P2    |          |
|LSTM                  |           |           |    P1    |          |
|LeakyRelu             |     +     |     +     |    P1    |    10    |
|Less                  |           |           |    ?     |          |
|Log                   |           |           |    ?     |          |
|LogSoftmax            |           |           |    ?     |          |
|Loop                  |           |           |    ?     |          |
|LpNormalization       |           |           |    ?     |          |
|LpPool                |           |           |    ?     |          |
|MatMul                |     +     |     +     |    P1    |     5    |
|Max                   |           |           |    ?     |          |
|MaxPool               |     +     |     +     |    P1    |     5    |
|MaxRoiPool            |           |           |    ?     |          |
|MaxUnpool             |           |           |    ?     |          |
|Mean                  |           |           |    ?     |          |
|Min                   |           |           |    ?     |          |
|Mul                   |     +     |     +     |    P1    |     5    |
|Mul-6                 |           |           |    P2    |          |
|Multinomial           |           |           |    ?     |          |
|Neg                   |     +     |           |    P2    |     5    |
|Not                   |           |           |    P3    |          |
|OneHot                |           |           |    P4    |          |
|Or                    |           |           |    ?     |          |
|PRelu                 |           |           |    P2    |          |
|Pad                   |           |           |    P1    |          |
|Pow                   |           |           |    P2    |          |
|RNN                   |           |           |    ?     |          |
|RandomNormal          |           |           |    ?     |          |
|RandomNormalLike      |           |           |    ?     |          |
|RandomUniform         |           |           |    ?     |          |
|RandomUniformLike     |           |           |    ?     |          |
|Reciprocal            |           |           |    ?     |          |
|ReduceL1              |           |           |    ?     |          |
|ReduceL2              |           |           |    ?     |          |
|ReduceLogSum          |           |           |    ?     |          |
|ReduceLogSumExp       |           |           |    ?     |          |
|ReduceMax             |           |           |    P2    |          |
|ReduceMean            |     +     |     +     |    P2    |     3    |
|ReduceMin             |           |           |    P2    |          |
|ReduceProd            |           |           |    P4    |          |
|ReduceSum             |           |           |    P3    |          |
|ReduceSumSquare       |           |           |    P4    |          |
|Relu                  |     +     |     +     |    P1    |    10    |
|Reshape               |     +     |     +     |    P1    |    10    |
|Scan                  |           |           |    ?     |          |
|Selu                  |           |           |    ?     |          |
|Shape                 |           |           |    P1    |          |
|Sigmoid               |     +     |     +     |    P1    |    10    |
|Sin                   |           |           |    ?     |          |
|Sinh                  |           |           |    ?     |          |
|Size                  |           |           |    ?     |          |
|Slice                 |           |           |    P2    |          |
|Softmax               |           |           |    P1    |          |
|Softplus              |           |           |    ?     |          |
|Softsign              |           |           |    ?     |          |
|SpaceToDepth          |           |           |    P3    |          |
|Split                 |           |           |    P1    |    10    |
|Sqrt                  |           |           |    P3    |          |
|Squeeze               |     +     |     +     |    P1    |    10    |
|Sub                   |     +     |     +     |    P3    |     5    |
|Sub-6                 |           |           |    P3    |          |
|Sum                   |           |           |    P3    |          |
|Tan                   |           |           |    P3    |          |
|Tanh                  |           |           |    P3    |          |
|Tile                  |           |           |    P3    |          |
|TopK                  |           |           |    P2    |          |
|Transpose             |           |           |    P1    |          |
|Unsqueeze             |     +     |     +     |    P1    |    10    |
|Upsample              |           |           |    P2    |          |
|Xor                   |           |           |    P3    |          |
|exp ATen              |           |           |    ?     |          |
|exp Affine            |           |           |    ?     |          |
|exp ConstantFill      |           |           |    P2    |          |
|exp Crop              |           |           |    P2    |          |
|exp DynamicSlice      |           |           |    ?     |          |
|exp GRUUnit           |           |           |    ?     |          |
|exp GivenTensorFill   |           |           |    ?     |          |
|exp ImageScaler       |           |           |    P2    |          |
|exp ParametricSoftplus|           |           |    ?     |          |
|exp Scale             |     +     |     +     |    P1    |    10    |
|exp ScaledTanh        |           |           |    ?     |          |
