// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnn_types.h"
#include <cstdint>
#include <cpp/ie_cnn_network.h>
#include <ie_algorithm.hpp>
#include <legacy/ie_layers.h>

namespace GNAPluginNS {
namespace GNALimitations {

constexpr uint32_t bufferMaxSize = 65528;

constexpr uint32_t convMinFiltersNum = 4;
constexpr uint32_t convMaxFiltersNum = 65532;
constexpr uint32_t convDilationHeight = 1;
constexpr uint32_t convDilationWidth = 1;
constexpr uint32_t convFiltersNumDivider = 4;
constexpr uint32_t convFilterSizeDivider = 8;
constexpr uint32_t convFilterMaxSize = 768;
constexpr uint32_t convEachKernelByteAlignment = 16;
constexpr uint32_t inputByteAlignment = 64;
constexpr uint32_t noOfInputsDivisor = 8;
constexpr uint32_t noOfInputsLowPrecDivisor = 16;

constexpr uint32_t affineMaxBatchSize = 8;

constexpr uint32_t maxPoolMaxWindowSize = 6;
constexpr uint32_t copyMaxGrouping = 8;
constexpr uint32_t transposeMaxSize = 65528;

inline bool IsTranspose2d(const std::vector<size_t>& shape) {
    return std::count_if(std::begin(shape), std::end(shape), [](size_t dim) { return dim != 1; }) == 2;
}

inline bool IsTransposeSupported(const std::vector<size_t>& shape) {
    if (!IsTranspose2d(shape)) return false;
    auto shape_no_1 = shape;
    shape_no_1.erase(std::remove(shape_no_1.begin(), shape_no_1.end(), 1), shape_no_1.end());
    size_t min, max;
    std::tie(min, max) = std::minmax(shape_no_1[0], shape_no_1[1]);
    return min <= 8 && max % 8 == 0 && max >= 8 && max <= transposeMaxSize;
}

namespace Cnn2D {
struct RangeLimit {
    uint32_t min;
    uint32_t max;
    std::string what;
    bool isValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct RangeLimit2D {
    RangeLimit hLimit;
    RangeLimit wLimit;
    bool isValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w) const;
};

struct RangeMultipleLimit : public RangeLimit {
    uint32_t multiplier;
    RangeMultipleLimit(RangeLimit rlIn, uint32_t multiplierIn);
    bool isValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct RectLimit {
    uint32_t maxVectorHeight;
    uint32_t maxVectorWidth;
    bool isValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const;
};

struct VectorOrSquareLimit {
    uint32_t maxSquare;
    uint32_t maxVectorHeight;
    uint32_t maxVectorWidth;
    bool isValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const;
};

struct RectLimitByChannels {
    std::vector<std::pair<uint32_t, RectLimit> > limitPerChannel;
    RectLimit GetByChannels(const uint32_t channels) const;
    bool isValid(const uint32_t h, const uint32_t w, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w,
        const uint32_t channels, std::string what) const;
};

struct RectLimitByChannelsAndPrecision {
    RectLimitByChannels lowPrecision;
    RectLimitByChannels defaultPrecision;
    RectLimitByChannels GetByPrecision(const OvGnaType precision) const;
    bool isValid(const uint32_t h, const uint32_t w, const OvGnaType precision, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w,
        const OvGnaType precision, const uint32_t channels, std::string what) const;
};

class Validator {
    RangeLimit2D inputHWLimit{ { 16, 384, "input height"} , { 16, 240, "input width"} };
    RangeMultipleLimit inputChannelsNumberLimit{ {8, 384, "number of input channels"}, 8 };

    RangeMultipleLimit kernelNumberLimit{ {8, 1024, "number of kernels"}, 8 };
    RectLimitByChannelsAndPrecision kernelLimit {
        { { {96, {7, 7}},
            {136, {7, 5}},
            {168, {7, 4}},
            {240, {7, 3}},
            {384, {7, 2}} } },
        { { {48, {7, 7}},
            {64, {7, 5}},
            {80, {7, 4}},
            {120, {7, 3}},
            {384, {7, 1}} } },
    };
    RectLimitByChannelsAndPrecision& strideLimit = kernelLimit;
    RangeLimit2D dilationLimit{ {convDilationHeight, convDilationHeight, "dilation height" },
        { convDilationWidth, convDilationWidth, "dilation width" } };
    const VectorOrSquareLimit poolingWindowLimit{ 3, 1, 1 };

    static void ThrowIfNotEmpty(const std::string prefix, const std::string error);

public:
    Validator() = default;

    bool ValidateCnn2D(std::string name, const uint32_t inHeight, const uint32_t inWidth,
        const uint32_t inChannels, const uint32_t kernelH, const uint32_t kernelW, const uint32_t kernelN,
        const uint32_t strideH, const uint32_t strideW, const uint32_t dilationH, const uint32_t dilationW,
        OvGnaType inPrecision, bool exception = true) const;

    bool ValidatePooling2D(std::string name,
        const uint32_t windowH, const uint32_t windowW,
        const uint32_t strideH, const uint32_t strideW,
        bool exception = true) const;
};
} // namespace Cnn2D

bool AreLayersSupported(InferenceEngine::CNNNetwork& network, std::string& errMessage, bool userWarning);

inline size_t GetMinBatchToFitInBuffer(InferenceEngine::DataPtr input) {
    auto total_size = InferenceEngine::details::product(std::begin(input->getDims()), std::end(input->getDims()));
    return total_size / bufferMaxSize + 1;
}

/**
 * @brief Validates if concat layer axis is supported by GNA
 * @param layer concat layer
 * @return true if concat layer axis is valid
 */
IE_SUPPRESS_DEPRECATED_START
bool ValidateConvConcatAxis(const InferenceEngine::ConcatLayer* concatLayer);
IE_SUPPRESS_DEPRECATED_END

} // namespace GNALimitations
} // namespace GNAPluginNS
