// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_layers.h>

using namespace InferenceEngine;

#if defined(__ANDROID__)

CNNLayer::~CNNLayer() {}

WeightableLayer::~WeightableLayer() {}

ConvolutionLayer::~ConvolutionLayer() {}

DeconvolutionLayer::~DeconvolutionLayer() {}

PoolingLayer::~PoolingLayer() {};

FullyConnectedLayer::~FullyConnectedLayer() {};

ConcatLayer::~ConcatLayer() {};

SplitLayer::~SplitLayer() {};

NormLayer::~NormLayer() {};

SoftMaxLayer::~SoftMaxLayer() {};

GRNLayer::~GRNLayer() {};

MVNLayer::~MVNLayer() {};

ReLULayer::~ReLULayer() {};

ClampLayer::~ClampLayer() {};

EltwiseLayer::~EltwiseLayer() {};

CropLayer::~CropLayer() {};

ReshapeLayer::~ReshapeLayer() {};

TileLayer::~TileLayer() {};

ScaleShiftLayer::~ScaleShiftLayer() {};

PReLULayer::~PReLULayer() {};

PowerLayer::~PowerLayer() {};

BatchNormalizationLayer::~BatchNormalizationLayer() {};

#endif
