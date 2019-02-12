/*
 * INTEL CONFIDENTIAL
 * Copyright 2016 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#include "ie_layers.h"
#include "ie_data.h"
#include <memory>
#include <string>
#include <map>

using namespace InferenceEngine;
#if defined(__ANDROID__)
CNNLayer::~CNNLayer() {

}

WeightableLayer::~WeightableLayer(){

}

ConvolutionLayer::~ConvolutionLayer(){

}

DeconvolutionLayer::~DeconvolutionLayer(){

}

PoolingLayer::~PoolingLayer(){

}

FullyConnectedLayer::~FullyConnectedLayer(){

}

ConcatLayer::~ConcatLayer(){

}

SplitLayer::~SplitLayer(){

}

NormLayer::~NormLayer(){

}

SoftMaxLayer::~SoftMaxLayer(){

}

ReLULayer::~ReLULayer(){

}

TanHLayer::~TanHLayer(){

}

SigmoidLayer::~SigmoidLayer(){

}

ClampLayer::~ClampLayer(){

}

EltwiseLayer::~EltwiseLayer(){

}

CropLayer::~CropLayer(){

}

ReshapeLayer::~ReshapeLayer(){

}

TileLayer::~TileLayer(){

}

ScaleShiftLayer::~ScaleShiftLayer(){

}

PowerLayer::~PowerLayer(){

}

BatchNormalizationLayer::~BatchNormalizationLayer(){

}

PReLULayer::~PReLULayer(){

}

GRNLayer::~GRNLayer(){

}

MVNLayer::~MVNLayer(){

}
ShuffleChannelsLayer::~ShuffleChannelsLayer() {}
DepthToSpaceLayer::~DepthToSpaceLayer() {}
SpaceToDepthLayer::~SpaceToDepthLayer() {}
GemmLayer::~GemmLayer() {}
PadLayer::~PadLayer() {}
GatherLayer::~GatherLayer() {}
TopKLayer::~TopKLayer() {}
UniqueLayer::~UniqueLayer() {}
NonMaxSuppressionLayer::~NonMaxSuppressionLayer() {}
SelectLayer::~SelectLayer() {}
ScatterLayer::~ScatterLayer() {}
SparseFillEmptyRowsLayer::~SparseFillEmptyRowsLayer() {}
SparseSegmentReduceLayer::~SparseSegmentReduceLayer() {}
ExperimentalSparseWeightedReduceLayer::~ExperimentalSparseWeightedReduceLayer() {}
SparseToDenseLayer::~SparseToDenseLayer() {}
BucketizeLayer::~BucketizeLayer() {}
ReverseSequenceLayer::~ReverseSequenceLayer() {}
OneHotLayer::~OneHotLayer() {}
RangeLayer::~RangeLayer() {}
TensorIterator::~TensorIterator() {}
BinaryConvolutionLayer::~BinaryConvolutionLayer() {}
RNNCell::~RNNCell() {}
LSTMCell::~LSTMCell() {}
GRUCell::~GRUCell() {}
FillLayer::~FillLayer() {}
BroadcastLayer::~BroadcastLayer() {}
QuantizeLayer::~QuantizeLayer() {}
MathLayer::~MathLayer() {}
ReduceLayer::~ReduceLayer() {}
ReLU6Layer::~ReLU6Layer() {}
RNNSequenceLayer::~RNNSequenceLayer() {}
RNNCellBase::~RNNCellBase() {}
DeformableConvolutionLayer::~DeformableConvolutionLayer() {}

#endif
