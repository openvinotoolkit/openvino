// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnn_types.h"
#include <cstdint>
#include <cpp/ie_cnn_network.h>
#include <ie_algorithm.hpp>

namespace GNAPluginNS {
namespace GNALimitations {

constexpr uint32_t bufferMaxSize = 65528;

constexpr uint32_t convMinFiltersNum = 4;
constexpr uint32_t convMaxFiltersNum = 65532;
constexpr uint32_t convFiltersNumDivider = 4;
constexpr uint32_t convFilterSizeDivider = 8;
constexpr uint32_t convFilterMaxSize = 768;
constexpr uint32_t convEachKernelByteAlignment = 16;
constexpr uint32_t noOfInputsDivisor = 8;
constexpr uint32_t noOfInputsLowPrecDivisor = 16;

constexpr uint32_t affineMaxBatchSize = 8;

constexpr uint32_t maxPoolMaxWindowSize = 6;
constexpr uint32_t copyMaxGrouping = 8;
constexpr uint32_t transposeMaxSize = 65528;

inline bool IsTransposeSupported(const std::vector<size_t>& shape) {
    auto shape_no_1 = shape;
    shape_no_1.erase(std::remove(shape_no_1.begin(), shape_no_1.end(), 1), shape_no_1.end());
    if (shape_no_1.size() != 2) return false;
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

struct VectorOrSquareLimit {
    uint32_t maxSquare;
    uint32_t maxVectorHeight;
    uint32_t maxVectorWidth;
    bool isValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const;
};

struct VectorOrSquareLimitByChannels {
    uint32_t smallChannelMax;
    VectorOrSquareLimit smallChannel;
    VectorOrSquareLimit bigChannel;
    VectorOrSquareLimit GetByChannels(const uint32_t channels) const;
    bool isValid(const uint32_t h, const uint32_t w, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w,
        const uint32_t channels, std::string what) const;
};

struct VectorOrSquareLimitByChannelsAndPrecision {
    VectorOrSquareLimitByChannels lowPrecision;
    VectorOrSquareLimitByChannels defaultPrecision;
    VectorOrSquareLimitByChannels GetByPrecision(const OvGnaType precision) const;
    bool isValid(const uint32_t h, const uint32_t w, const OvGnaType precision, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w,
        const OvGnaType precision, const uint32_t channels, std::string what) const;
};

class Validator {
    RangeLimit2D inputHWLimit{ { 16, 384, "input height"} , { 16, 240, "input width"} };
    RangeMultipleLimit inputChannelsNumberLimit{ {8, 384, "number of input channels"}, 8 };

    RangeMultipleLimit kernelNumberLimit{ {8, 256, "number of kernels"}, 8 };
    VectorOrSquareLimitByChannelsAndPrecision kernelLimit {
        { 240, { 3, 7, 3 }, { 2, 7, 2 } },
        { 120, { 3, 7, 3 }, { 1, 7, 1 } } };
    VectorOrSquareLimitByChannelsAndPrecision& strideLimit = kernelLimit;
    const VectorOrSquareLimit poolingWindowLimit{ 3, 1, 1 };

    static void ThrowIfNotEmpty(const std::string prefix, const std::string error);
public:
    Validator() = default;

    void ValidateCnn2D(std::string name, const uint32_t inHeight, const uint32_t inWidth,
        const uint32_t inChannels, const uint32_t kH, const uint32_t kW, const uint32_t kN,
        const uint32_t strideH, const uint32_t strideW, OvGnaType inPrecision) const;

    void ValidatePooling2D(std::string name,
        const uint32_t windowH, const uint32_t windowW,
        const uint32_t strideH, const uint32_t strideW) const;
};
} // namespace Cnn2D

bool AreLayersSupported(InferenceEngine::CNNNetwork& network, std::string& errMessage);

inline size_t GetMinBatchToFitInBuffer(InferenceEngine::DataPtr input) {
    auto total_size = InferenceEngine::details::product(std::begin(input->getDims()), std::end(input->getDims()));
    return total_size / bufferMaxSize + 1;
}

} // namespace GNALimitations
} // namespace GNAPluginNS
