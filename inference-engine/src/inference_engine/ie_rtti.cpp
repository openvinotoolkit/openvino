// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <ie_layers.h>

#include <details/ie_exception.hpp>
#include <ie_parameter.hpp>
#include <string>
#include <tuple>
#include <vector>

using namespace InferenceEngine;

//
// details/ie_exception.hpp
//
details::InferenceEngineException::~InferenceEngineException() noexcept {}
//
// ie_layers.h
//
CNNLayer::~CNNLayer() {}
WeightableLayer::~WeightableLayer() {}
ConvolutionLayer::~ConvolutionLayer() {}
DeconvolutionLayer::~DeconvolutionLayer() {}
DeformableConvolutionLayer::~DeformableConvolutionLayer() {}
PoolingLayer::~PoolingLayer() {}
BinaryConvolutionLayer::~BinaryConvolutionLayer() {}
FullyConnectedLayer::~FullyConnectedLayer() {}
ConcatLayer::~ConcatLayer() {}
SplitLayer::~SplitLayer() {}
NormLayer::~NormLayer() {}
SoftMaxLayer::~SoftMaxLayer() {}
GRNLayer::~GRNLayer() {}
MVNLayer::~MVNLayer() {}
ReLULayer::~ReLULayer() {}
ClampLayer::~ClampLayer() {}
ReLU6Layer::~ReLU6Layer() {}
EltwiseLayer::~EltwiseLayer() {}
CropLayer::~CropLayer() {}
ReshapeLayer::~ReshapeLayer() {}
TileLayer::~TileLayer() {}
ScaleShiftLayer::~ScaleShiftLayer() {}
TensorIterator::~TensorIterator() {}
RNNCellBase::~RNNCellBase() {}
LSTMCell::~LSTMCell() {}
GRUCell::~GRUCell() {}
RNNCell::~RNNCell() {}
RNNSequenceLayer::~RNNSequenceLayer() {}
PReLULayer::~PReLULayer() {}
PowerLayer::~PowerLayer() {}
BatchNormalizationLayer::~BatchNormalizationLayer() {}
GemmLayer::~GemmLayer() {}
PadLayer::~PadLayer() {}
GatherLayer::~GatherLayer() {}
StridedSliceLayer::~StridedSliceLayer() {}
ShuffleChannelsLayer::~ShuffleChannelsLayer() {}
DepthToSpaceLayer::~DepthToSpaceLayer() {}
SpaceToDepthLayer::~SpaceToDepthLayer() {}
SparseFillEmptyRowsLayer::~SparseFillEmptyRowsLayer() {}
SparseSegmentReduceLayer::~SparseSegmentReduceLayer() {}
ExperimentalSparseWeightedReduceLayer::~ExperimentalSparseWeightedReduceLayer() {}
SparseToDenseLayer::~SparseToDenseLayer() {}
BucketizeLayer::~BucketizeLayer() {}
ReverseSequenceLayer::~ReverseSequenceLayer() {}
OneHotLayer::~OneHotLayer() {}
RangeLayer::~RangeLayer() {}
FillLayer::~FillLayer() {}
SelectLayer::~SelectLayer() {}
BroadcastLayer::~BroadcastLayer() {}
QuantizeLayer::~QuantizeLayer() {}
MathLayer::~MathLayer() {}
ReduceLayer::~ReduceLayer() {}
TopKLayer::~TopKLayer() {}
UniqueLayer::~UniqueLayer() {}
NonMaxSuppressionLayer::~NonMaxSuppressionLayer() {}
ScatterLayer::~ScatterLayer() {}
//
// ie_parameter.hpp
//
Parameter::~Parameter() {
    clear();
}

#ifdef __clang__
Parameter::Any::~Any() {}

template struct InferenceEngine::Parameter::RealData<int>;
template struct InferenceEngine::Parameter::RealData<bool>;
template struct InferenceEngine::Parameter::RealData<float>;
template struct InferenceEngine::Parameter::RealData<uint32_t>;
template struct InferenceEngine::Parameter::RealData<std::string>;
template struct InferenceEngine::Parameter::RealData<unsigned long>;
template struct InferenceEngine::Parameter::RealData<std::vector<int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::string>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned long>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>;
#endif  // __clang__
//
// ie_blob.h
//
Blob::~Blob() {}

MemoryBlob::~MemoryBlob() {}

#ifdef __clang__
template <typename T, typename U>
TBlob<T, U>::~TBlob() {
    free();
}

template class InferenceEngine::TBlob<float>;
template class InferenceEngine::TBlob<double>;
template class InferenceEngine::TBlob<int16_t>;
template class InferenceEngine::TBlob<uint16_t>;
template class InferenceEngine::TBlob<int8_t>;
template class InferenceEngine::TBlob<uint8_t>;
template class InferenceEngine::TBlob<int>;
template class InferenceEngine::TBlob<long>;
template class InferenceEngine::TBlob<long long>;
#endif  // __clang__
