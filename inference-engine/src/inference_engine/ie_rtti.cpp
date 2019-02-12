// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <ie_layers.h>

#include <details/ie_exception.hpp>
#include <ie_parameter.hpp>
#include <inference_engine.hpp>
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
SigmoidLayer::~SigmoidLayer() {}
TanHLayer::~TanHLayer() {}
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
//
// inference_engine.hpp
//
#define TBLOB_TOP_RESULT(precision)                                                           \
    case InferenceEngine::Precision::precision: {                                             \
        using myBlobType = InferenceEngine::PrecisionTrait<Precision::precision>::value_type; \
        TBlob<myBlobType>& tblob = dynamic_cast<TBlob<myBlobType>&>(input);                   \
        TopResults(n, tblob, output);                                                         \
        break;                                                                          \
	}

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Gets the top n results from a blob
 *
 * @param n Top n count
 * @param input 1D blob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
void TopResults(unsigned int n, Blob& input, std::vector<unsigned>& output) {
    IE_SUPPRESS_DEPRECATED_START
    switch (input.getTensorDesc().getPrecision()) {
        TBLOB_TOP_RESULT(FP32);
        TBLOB_TOP_RESULT(FP16);
        TBLOB_TOP_RESULT(Q78);
        TBLOB_TOP_RESULT(I16);
        TBLOB_TOP_RESULT(U8);
        TBLOB_TOP_RESULT(I8);
        TBLOB_TOP_RESULT(U16);
        TBLOB_TOP_RESULT(I32);
    default:
        THROW_IE_EXCEPTION << "cannot locate blob for precision: " << input.getTensorDesc().getPrecision();
    }
    IE_SUPPRESS_DEPRECATED_END
}

#undef TBLOB_TOP_RESULT

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 *
 * The image buffer is assumed to be packed with no support for strides.
 *
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
void ConvertImageToInput(unsigned char* imgBufRGB8, size_t lengthbytesSize, Blob& input) {
    IE_SUPPRESS_DEPRECATED_START
    TBlob<float>* float_input = dynamic_cast<TBlob<float>*>(&input);
    if (float_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, float_input);

    TBlob<short>* short_input = dynamic_cast<TBlob<short>*>(&input);
    if (short_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, short_input);

    TBlob<uint8_t>* byte_input = dynamic_cast<TBlob<uint8_t>*>(&input);
    if (byte_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, byte_input);
    IE_SUPPRESS_DEPRECATED_END
}
