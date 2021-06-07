// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_limitations.hpp"

#include <cstdint>

using GNAPluginNS::GNALimitations::Cnn2D::Validator;
using GNAPluginNS::GNALimitations::Cnn2D::VectorOrSquareLimit;
using GNAPluginNS::GNALimitations::Cnn2D::VectorOrSquareLimitByChannels;
using GNAPluginNS::GNALimitations::Cnn2D::VectorOrSquareLimitByChannelsAndPrecision;
using GNAPluginNS::GNALimitations::Cnn2D::RangeLimit;
using GNAPluginNS::GNALimitations::Cnn2D::RangeLimit2D;
using GNAPluginNS::GNALimitations::Cnn2D::RangeMultipleLimit;

bool RangeLimit::isValid(const uint32_t val) const {
    return val >= min && val <= max;
}

std::string RangeLimit::GetErrorOrEmpty(const uint32_t val) const {
    std::ostringstream out;
    if (!isValid(val)) {
        out << "Unsupported " << what << ", actual value: " << val << ", valid range [" << min << ", " << max << "]\n";
    }
    return out.str();
}

bool RangeLimit2D::isValid(const uint32_t h, const uint32_t w) const {
    return hLimit.isValid(h) && wLimit.isValid(w);
}

std::string RangeLimit2D::GetErrorOrEmpty(const uint32_t h, const uint32_t w) const {
    return hLimit.GetErrorOrEmpty(h) + hLimit.GetErrorOrEmpty(w);
}

RangeMultipleLimit::RangeMultipleLimit(RangeLimit rlIn, uint32_t multiplierIn) : RangeLimit(rlIn), multiplier(multiplierIn) {
}

bool RangeMultipleLimit::isValid(const uint32_t val) const {
    return RangeLimit::isValid(val) && (val % multiplier == 0);
}

std::string RangeMultipleLimit::GetErrorOrEmpty(const uint32_t val) const {
    auto e = RangeLimit::GetErrorOrEmpty(val);
    std::ostringstream out;
    if (val % multiplier != 0) {
        out << "Unsupported " << what << ": " << val << ", must be multiple of " << multiplier << "\n";
    }
    return e + out.str();
}

bool VectorOrSquareLimit::isValid(const uint32_t h, const uint32_t w) const {
    if (w == 1 && h >= 1 && h <= maxVectorHeight) return true;
    if (h == 1 && w >= 1 && w <= maxVectorWidth) return true;
    if (h == w && h <= maxSquare && h >= 1) return true;
    return false;
}

std::string VectorOrSquareLimit::GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const {
    std::ostringstream out;
    if (!isValid(h, w)) {
        out << "Unsupported " << what << " shape, actual WxH: " << w << "x" << h <<
            ", only vertical vector up to 1x" << maxVectorHeight << ", horizontal up to " << maxVectorWidth <<
            "x1 or square up to " << maxSquare << "x" << maxSquare << " are valid\n";
    }
    return out.str();
}

VectorOrSquareLimit VectorOrSquareLimitByChannels::GetByChannels(const uint32_t channels) const {
    return channels <= smallChannelMax ? smallChannel : bigChannel;
}

bool VectorOrSquareLimitByChannels::isValid(const uint32_t h, const uint32_t w, const uint32_t channels) const {
    return GetByChannels(channels).isValid(h, w);
}

std::string VectorOrSquareLimitByChannels::GetErrorOrEmpty(const uint32_t h, const uint32_t w,
    const uint32_t channels, std::string what) const {
    return GetByChannels(channels).GetErrorOrEmpty(h, w, what);
}

VectorOrSquareLimitByChannels VectorOrSquareLimitByChannelsAndPrecision::GetByPrecision(const OvGnaType precision) const {
    return precision == OvGnaTypeInt8 ? lowPrecision : defaultPrecision;
}

bool VectorOrSquareLimitByChannelsAndPrecision::isValid(const uint32_t h, const uint32_t w, const OvGnaType precision, const uint32_t channels) const {
    return GetByPrecision(precision).isValid(h, w, channels);
}

std::string VectorOrSquareLimitByChannelsAndPrecision::GetErrorOrEmpty(const uint32_t h, const uint32_t w,
    const OvGnaType precision, const uint32_t channels, std::string what) const {
    return GetByPrecision(precision).GetErrorOrEmpty(h, w, channels, what);
}

void Validator::ValidateCnn2D(std::string name, const uint32_t inHeight, const uint32_t inWidth,
    const uint32_t inChannels, const uint32_t kH, const uint32_t kW, const uint32_t kN,
    const uint32_t strideH, const uint32_t strideW, OvGnaType inPrecision) const {
    const std::string prefix = "Layer Convolution2D: " + name + ":";
    auto error = inputHWLimit.GetErrorOrEmpty(inHeight, inWidth);

    error += kernelNumberLimit.GetErrorOrEmpty(kN);

    error += inputChannelsNumberLimit.GetErrorOrEmpty(inChannels);
    error += kernelLimit.GetErrorOrEmpty(kH, kW, inPrecision, inChannels, "kernel");
    error += strideLimit.GetErrorOrEmpty(strideH, strideW, inPrecision, inChannels, "convolution stride");
    ThrowIfNotEmpty(prefix, error);
}

void Validator::ValidatePooling2D(std::string name,
    const uint32_t windowH, const uint32_t windowW,
    const uint32_t strideH, const uint32_t strideW) const {
    const std::string prefix = "Layer Pooling2D: " + name + ":";

    auto error = poolingWindowLimit.GetErrorOrEmpty(windowH, windowW, "pooling window");
    const RangeLimit poolingStrideHLimit{ 1, windowH, "pooling stride height (must be up to pooling window height)" };
    const RangeLimit poolingStrideWLimit{ 1, windowW, "pooling stride width (must be up to pooling window width)" };

    error += poolingStrideHLimit.GetErrorOrEmpty(strideH);
    error += poolingStrideWLimit.GetErrorOrEmpty(strideW);

    ThrowIfNotEmpty(prefix, error);
}

void Validator::ThrowIfNotEmpty(const std::string prefix, const std::string error) {
    if (!error.empty()) {
        THROW_GNA_EXCEPTION << prefix << error;
    }
}
