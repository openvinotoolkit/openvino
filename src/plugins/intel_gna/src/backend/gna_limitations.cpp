// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gna_limitations.hpp"

#include <legacy/ie_layers.h>

#include <cstdint>
#include <layers/gna_layer_info.hpp>
#include <layers/gna_layer_type.hpp>
#include <legacy/graph_tools.hpp>
#include <unordered_set>

#include "common/gna_target.hpp"
#include "common/graph_utils.hpp"
#include "gna/gna_config.hpp"
#include "gna_graph_tools.hpp"
#include "gna_lib_ver_selector.hpp"
#include "ie_ngraph_utils.hpp"
#include "log/log.hpp"
#include "openvino/opsets/opset12.hpp"

namespace std {
inline std::ostream& operator<<(std::ostream& os, const std::set<ov::element::Type>& t) {
    for (auto it = t.begin(); it != t.end(); ++it) {
        if (it != t.begin()) {
            os << ", " << *it;
        } else {
            os << *it;
        }
    }
    return os;
}
}  // namespace std

namespace ov {
namespace intel_gna {
using namespace target;
using namespace opset12;
namespace limitations {

class SupportedElementTypes {
public:
    static bool IsParameterTypeSupported(ov::element::Type type, bool is_exception_allowed = false);
    static bool IsConstantTypeSupported(ov::element::Type type, bool is_exception_allowed = false);

private:
    static const std::set<ov::element::Type> supported_parameter_types;
    static const std::set<ov::element::Type> supported_constant_types;
};

const std::set<ov::element::Type> SupportedElementTypes::supported_parameter_types = {ov::element::u8,
                                                                                      ov::element::i16,
                                                                                      ov::element::f32};

namespace cnn2d {

bool IsEqualToLimit::IsValid(const uint32_t val) const {
    return val == compared_value;
}

std::string IsEqualToLimit::GetErrorOrEmpty(const uint32_t val) const {
    std::ostringstream out;
    if (!IsValid(val)) {
        out << "Unsupported " << what << ", actual value: " << val << ", but should be equal to " << compared_value
            << "\n";
    }
    return out.str();
}

bool IsLessThanLimit::IsValid(const uint32_t val) const {
    return val < compared_value;
}

std::string IsLessThanLimit::GetErrorOrEmpty(const uint32_t val) const {
    std::ostringstream out;
    if (!IsValid(val)) {
        out << "Unsupported " << what << ", actual value: " << val << ", but should be less than " << compared_value
            << "\n";
    }
    return out.str();
}

bool RangeLimit::IsValid(const uint32_t val) const {
    return val >= min && val <= max;
}

std::string RangeLimit::GetErrorOrEmpty(const uint32_t val) const {
    std::ostringstream out;
    if (!IsValid(val)) {
        out << "Unsupported " << what << ", actual value: " << val << ", valid range [" << min << ", " << max << "]\n";
    }
    return out.str();
}

bool RangeLimit2D::IsValid(const uint32_t h, const uint32_t w) const {
    return hLimit.IsValid(h) && wLimit.IsValid(w);
}

std::string RangeLimit2D::GetErrorOrEmpty(const uint32_t h, const uint32_t w) const {
    return hLimit.GetErrorOrEmpty(h) + wLimit.GetErrorOrEmpty(w);
}

RangeMultipleLimit::RangeMultipleLimit(RangeLimit rlIn, uint32_t multiplierIn)
    : RangeLimit(rlIn),
      multiplier(multiplierIn) {}

bool RangeMultipleLimit::IsValid(const uint32_t val) const {
    return RangeLimit::IsValid(val) && (val % multiplier == 0);
}

std::string RangeMultipleLimit::GetErrorOrEmpty(const uint32_t val) const {
    auto e = RangeLimit::GetErrorOrEmpty(val);
    std::ostringstream out;
    if (val % multiplier != 0) {
        out << "Unsupported " << what << ": " << val << ", must be multiple of " << multiplier << "\n";
    }
    return e + out.str();
}

bool VectorOrSquareLimit::IsValid(const uint32_t h, const uint32_t w) const {
    if (w == 1 && h >= 1 && h <= maxVectorHeight)
        return true;
    if (h == 1 && w >= 1 && w <= maxVectorWidth)
        return true;
    if (h == w && h <= maxSquare && h >= 1)
        return true;
    return false;
}

std::string VectorOrSquareLimit::GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const {
    std::ostringstream out;
    if (!IsValid(h, w)) {
        out << "Unsupported " << what << " shape, actual HxW: " << h << "x" << w << ", only vertical vector up to "
            << maxVectorHeight << "x1, horizontal up to 1x" << maxVectorWidth << " or square up to " << maxSquare << "x"
            << maxSquare << " are valid\n";
    }
    return out.str();
}

bool RectLimit::IsValid(const uint32_t h, const uint32_t w) const {
    if (h >= 1 && h <= maxVectorHeight && w >= 1 && w <= maxVectorWidth)
        return true;
    return false;
}

std::string RectLimit::GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const {
    std::ostringstream out;
    if (!IsValid(h, w)) {
        out << "Unsupported " << what << " shape, actual HxW: " << h << "x" << w << ", only rectangular shapes up to "
            << maxVectorHeight << "x" << maxVectorWidth << " are valid\n";
    }
    return out.str();
}

RectLimit RectLimitByChannels::GetByChannels(const uint32_t channels) const {
    for (auto&& limit : limitPerChannel) {
        if (limit.first >= channels) {
            return limit.second;
        }
    }
    return RectLimit{0, 0};
}

bool RectLimitByChannels::IsValid(const uint32_t h, const uint32_t w, const uint32_t channels) const {
    return GetByChannels(channels).IsValid(h, w);
}

std::string RectLimitByChannels::GetErrorOrEmpty(const uint32_t h,
                                                 const uint32_t w,
                                                 const uint32_t channels,
                                                 std::string what) const {
    return GetByChannels(channels).GetErrorOrEmpty(h, w, what);
}

RectLimitByChannels RectLimitByChannelsAndPrecision::GetByPrecision(const OvGnaType precision) const {
    return precision == OvGnaTypeInt8 ? limit_for_int8 : limit_for_int16;
}

bool RectLimitByChannelsAndPrecision::IsValid(const uint32_t h,
                                              const uint32_t w,
                                              const OvGnaType precision,
                                              const uint32_t channels) const {
    return GetByPrecision(precision).IsValid(h, w, channels);
}

std::string RectLimitByChannelsAndPrecision::GetErrorOrEmpty(const uint32_t h,
                                                             const uint32_t w,
                                                             const OvGnaType precision,
                                                             const uint32_t channels,
                                                             std::string what) const {
    return GetByPrecision(precision).GetErrorOrEmpty(h, w, channels, what);
}

class Validator_30 : public AbstractValidator {
    static const RangeLimit2D kInputHWLimit;
    static const RangeMultipleLimit kInputChannelsNumberLimit;

    static const RangeMultipleLimit kKernelNumberLimit;
    static const RectLimitByChannelsAndPrecision kKernelLimit;
    static const RangeLimit2D kDilationLimit;

    static const VectorOrSquareLimit kPoolingWindowLimit;

public:
    Validator_30() = default;

    bool ValidateCnn2D(const std::string& name,
                       const uint32_t inHeight,
                       const uint32_t inWidth,
                       const uint32_t inChannels,
                       const uint32_t kH,
                       const uint32_t kW,
                       const uint32_t kN,
                       const uint32_t strideH,
                       const uint32_t strideW,
                       const uint32_t dilationH,
                       const uint32_t dilationW,
                       OvGnaType inPrecision,
                       bool exception = true) const override;

    bool ValidatePooling2D(const std::string& name,
                           const uint32_t windowH,
                           const uint32_t windowW,
                           const uint32_t strideH,
                           const uint32_t strideW,
                           bool exception = true) const override;

    bool ValidateInputPadding(const std::string& name,
                              const uint32_t pad_h_begin,
                              const uint32_t pad_h_end,
                              const uint32_t pad_w_begin,
                              const uint32_t pad_w_end,
                              const uint32_t kernel_h,
                              const uint32_t kernel_w,
                              const bool throwOnError = true) const override;

    bool ShouldUseOnlyConv2DGnaIface() const override;

    bool ValidateCnn1D(const std::string& name,
                       const uint32_t inHeight,
                       const uint32_t inWidth,
                       const uint32_t inChannels,
                       const uint32_t kH,
                       const uint32_t kW,
                       const uint32_t kN,
                       const uint32_t strideH,
                       const uint32_t strideW,
                       const uint32_t dilationH,
                       const uint32_t dilationW,
                       OvGnaType inPrecision,
                       bool exception = true) const override;
};

const RangeLimit2D Validator_30::kInputHWLimit{{16, 384, "input height"}, {16, 240, "input width"}};
const RangeMultipleLimit Validator_30::kInputChannelsNumberLimit{{8, 384, "number of input channels"}, 8};

const RangeMultipleLimit Validator_30::kKernelNumberLimit{{8, 1024, "number of kernels"}, 8};
const RectLimitByChannelsAndPrecision Validator_30::kKernelLimit{
    {{{96, {7, 7}}, {136, {7, 5}}, {168, {7, 4}}, {240, {7, 3}}, {384, {7, 2}}}},
    {{{48, {7, 7}}, {64, {7, 5}}, {80, {7, 4}}, {120, {7, 3}}, {384, {7, 1}}}},
};

const RangeLimit2D Validator_30::kDilationLimit{
    {Limitations::kConvDilationHeight, Limitations::kConvDilationHeight, "dilation height"},
    {Limitations::kConvDilationWidth, Limitations::kConvDilationWidth, "dilation width"}};

bool Validator_30::ValidateCnn2D(const std::string& name,
                                 const uint32_t inHeight,
                                 const uint32_t inWidth,
                                 const uint32_t inChannels,
                                 const uint32_t kernelH,
                                 const uint32_t kernelW,
                                 const uint32_t kernelN,
                                 const uint32_t strideH,
                                 const uint32_t strideW,
                                 const uint32_t dilationH,
                                 const uint32_t dilationW,
                                 const OvGnaType inPrecision,
                                 const bool throwOnError) const {
    const auto& kStrideLimit = kKernelLimit;
    auto error = kInputHWLimit.GetErrorOrEmpty(inHeight, inWidth);

    error += kKernelNumberLimit.GetErrorOrEmpty(kernelN);
    error += kInputChannelsNumberLimit.GetErrorOrEmpty(inChannels);
    error += kKernelLimit.GetErrorOrEmpty(kernelH, kernelW, inPrecision, inChannels, "kernel");
    error += kStrideLimit.GetErrorOrEmpty(strideH, strideW, inPrecision, inChannels, "convolution stride");

    const RangeLimit kKernelStrideHLimit{1, kernelH, "kernel stride height (must be up to kernel height)"};
    const RangeLimit kKernelStrideWLimit{1, kernelW, "kernel stride width (must be up to kernel width)"};

    error += kKernelStrideHLimit.GetErrorOrEmpty(strideH);
    error += kKernelStrideWLimit.GetErrorOrEmpty(strideW);

    error += kDilationLimit.GetErrorOrEmpty(dilationH, dilationW);

    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

bool Validator_30::ValidateCnn1D(const std::string& name,
                                 const uint32_t inHeight,
                                 const uint32_t inWidth,
                                 const uint32_t inChannels,
                                 const uint32_t kH,
                                 const uint32_t kW,
                                 const uint32_t kN,
                                 const uint32_t strideH,
                                 const uint32_t strideW,
                                 const uint32_t dilationH,
                                 const uint32_t dilationW,
                                 OvGnaType inPrecision,
                                 bool exception) const {
    return false;
}

const VectorOrSquareLimit Validator_30::kPoolingWindowLimit{3, 1, 1};

bool Validator_30::ValidatePooling2D(const std::string& name,
                                     const uint32_t windowH,
                                     const uint32_t windowW,
                                     const uint32_t strideH,
                                     const uint32_t strideW,
                                     const bool throwOnError) const {
    auto error = kPoolingWindowLimit.GetErrorOrEmpty(windowH, windowW, "pooling window");
    const RangeLimit poolingStrideHLimit{1, windowH, "pooling stride height (must be up to pooling window height)"};
    const RangeLimit poolingStrideWLimit{1, windowW, "pooling stride width (must be up to pooling window width)"};

    error += poolingStrideHLimit.GetErrorOrEmpty(strideH);
    error += poolingStrideWLimit.GetErrorOrEmpty(strideW);

    return ValidationSuccesful(throwOnError, error, name, "Pooling2D");
}

bool Validator_30::ValidateInputPadding(const std::string& name,
                                        const uint32_t pad_h_begin,
                                        const uint32_t pad_h_end,
                                        const uint32_t pad_w_begin,
                                        const uint32_t pad_w_end,
                                        const uint32_t,
                                        const uint32_t,
                                        const bool throwOnError) const {
    const IsEqualToLimit padding_zero{0, "convolution input padding size (must equal zero)"};
    auto error = padding_zero.GetErrorOrEmpty(pad_h_begin);
    error += padding_zero.GetErrorOrEmpty(pad_h_end);
    error += padding_zero.GetErrorOrEmpty(pad_w_begin);
    error += padding_zero.GetErrorOrEmpty(pad_w_end);
    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

bool Validator_30::ShouldUseOnlyConv2DGnaIface() const {
    return false;
}

class Validator_35 : public AbstractValidator {
    struct CnnLimits {
        const RangeLimit2D kInputHWLimit;
        const RangeLimit kInputChannelsNumberLimit1B;
        const RangeLimit kInputChannelsNumberLimit2B;
        const RangeLimit kKernelNumberLimit;
        const RangeLimit2D kKerneHWlLimit1B;
        const RangeLimit2D kKerneHWlLimit2B;
        const RangeLimit2D kStrideHWLimit1B;
        const RangeLimit2D kStrideHWLimit2B;
        const RangeLimit2D kDilationLimit;
        const RangeLimit2D kPoolingWindowHWLimit;
        const RangeLimit2D kPoolingStrideHWLimit;
    };

    static const CnnLimits kCnn2DLimits;
    static const CnnLimits kCnn1DLimits;

    std::string ValidateCnn(const CnnLimits& limits,
                            const std::string& name,
                            const uint32_t inHeight,
                            const uint32_t inWidth,
                            const uint32_t inChannels,
                            const uint32_t kH,
                            const uint32_t kW,
                            const uint32_t kN,
                            const uint32_t strideH,
                            const uint32_t strideW,
                            const uint32_t dilationH,
                            const uint32_t dilationW,
                            OvGnaType inPrecision) const;

    std::string ValidatePooling(const CnnLimits& limits,
                                const std::string& name,
                                const uint32_t windowH,
                                const uint32_t windowW,
                                const uint32_t strideH,
                                const uint32_t strideW) const;

public:
    Validator_35() = default;

    bool ValidateCnn2D(const std::string& name,
                       const uint32_t inHeight,
                       const uint32_t inWidth,
                       const uint32_t inChannels,
                       const uint32_t kH,
                       const uint32_t kW,
                       const uint32_t kN,
                       const uint32_t strideH,
                       const uint32_t strideW,
                       const uint32_t dilationH,
                       const uint32_t dilationW,
                       OvGnaType inPrecision,
                       bool exception = true) const override;

    bool ValidatePooling2D(const std::string& name,
                           const uint32_t windowH,
                           const uint32_t windowW,
                           const uint32_t strideH,
                           const uint32_t strideW,
                           bool exception = true) const override;

    bool ValidateInputPadding(const std::string& name,
                              const uint32_t pad_h_begin,
                              const uint32_t pad_h_end,
                              const uint32_t pad_w_begin,
                              const uint32_t pad_w_end,
                              const uint32_t kernel_h,
                              const uint32_t kernel_w,
                              const bool throwOnError = true) const override;

    bool ShouldUseOnlyConv2DGnaIface() const override;

    bool ValidateCnn1D(const std::string& name,
                       const uint32_t inHeight,
                       const uint32_t inWidth,
                       const uint32_t inChannels,
                       const uint32_t kH,
                       const uint32_t kW,
                       const uint32_t kN,
                       const uint32_t strideH,
                       const uint32_t strideW,
                       const uint32_t dilationH,
                       const uint32_t dilationW,
                       OvGnaType inPrecision,
                       bool exception = true) const override;
};

const Validator_35::CnnLimits Validator_35::kCnn2DLimits{
    {{1, 65535, "input height"}, {1, 65535, "input width"}},                        // kInputHWLimit
    {1, 2048, "number of input channels"},                                          // kInputChannelsNumberLimit1B
    {1, 1024, "number of input channels"},                                          // kInputChannelsNumberLimit2B
    {1, 8192, "number of kernels"},                                                 // kKernelNumberLimit
    {{1, 255, "kernel height"}, {1, 256, "kernel width"}},                          // kKerneHWlLimit1B
    {{1, 255, "kernel height"}, {1, 256, "kernel width"}},                          // kKerneHWlLimit2B
    {{1, 255, "convolution stride height"}, {1, 256, "convolution stride width"}},  // kStrideHWLimit1B
    {{1, 255, "convolution stride height"}, {1, 256, "convolution stride width"}},  // kStrideHWLimit2B
    {{Limitations::kConvDilationHeight, Limitations::kConvDilationHeight, "dilation height"},  // kDilationLimit
     {Limitations::kConvDilationWidth, Limitations::kConvDilationWidth, "dilation width"}},
    {{1, 255, "pooling window height"}, {1, 255, "pooling window width"}},  // kPoolingWindowHWLimit
    {{1, 255, "pooling stride height"}, {1, 255, "pooling stride width"}}   // kPoolingStrideHWLimit
};

const Validator_35::CnnLimits Validator_35::kCnn1DLimits{
    {{1, 1, "input height"}, {1, 65535, "input width"}},                           // kInputHWLimit
    {1, 1, "number of input channels"},                                            // kInputChannelsNumberLimit1B
    {1, 1, "number of input channels"},                                            // kInputChannelsNumberLimit2B
    {1, 8192, "number of kernels"},                                                // kKernelNumberLimit
    {{1, 1, "kernel height"}, {1, 4096, "kernel width"}},                          // kKerneHWlLimit1B
    {{1, 1, "kernel height"}, {1, 2048, "kernel width"}},                          // kKerneHWlLimit2B
    {{1, 1, "convolution stride height"}, {1, 4096, "convolution stride width"}},  // kStrideHWLimit1B
    {{1, 1, "convolution stride height"}, {1, 2048, "convolution stride width"}},  // kStrideHWLimit2B
    {{Limitations::kConvDilationHeight, Limitations::kConvDilationHeight, "dilation height"},  // kDilationLimit
     {Limitations::kConvDilationWidth, Limitations::kConvDilationWidth, "dilation width"}},
    {{1, 1, "pooling window height"}, {1, 255, "pooling window width"}},  // kPoolingWindowHWLimit
    {{1, 1, "pooling stride height"}, {1, 255, "pooling stride width"}}   // kPoolingStrideHWLimit
};

std::string Validator_35::ValidateCnn(const Validator_35::CnnLimits& limits,
                                      const std::string& name,
                                      const uint32_t inHeight,
                                      const uint32_t inWidth,
                                      const uint32_t inChannels,
                                      const uint32_t kernelH,
                                      const uint32_t kernelW,
                                      const uint32_t kernelN,
                                      const uint32_t strideH,
                                      const uint32_t strideW,
                                      const uint32_t dilationH,
                                      const uint32_t dilationW,
                                      const OvGnaType inPrecision) const {
    auto error = limits.kInputHWLimit.GetErrorOrEmpty(inHeight, inWidth);
    error += limits.kKernelNumberLimit.GetErrorOrEmpty(kernelN);
    auto& inputChannelsNumberLimit =
        (inPrecision == OvGnaTypeInt8) ? limits.kInputChannelsNumberLimit1B : limits.kInputChannelsNumberLimit2B;
    error += inputChannelsNumberLimit.GetErrorOrEmpty(inChannels);
    auto& kerneHWlLimit = (inPrecision == OvGnaTypeInt8) ? limits.kKerneHWlLimit1B : limits.kKerneHWlLimit2B;
    error += kerneHWlLimit.GetErrorOrEmpty(kernelH, kernelW);
    auto& strideHWLimit = (inPrecision == OvGnaTypeInt8) ? limits.kStrideHWLimit1B : limits.kStrideHWLimit2B;
    error += strideHWLimit.GetErrorOrEmpty(strideH, strideW);

    const RangeLimit kKernelStrideHLimit{1, kernelH, "kernel stride height (must be up to kernel height)"};
    const RangeLimit kKernelStrideWLimit{1, kernelW, "kernel stride width (must be up to kernel width)"};

    error += kKernelStrideHLimit.GetErrorOrEmpty(strideH);
    error += kKernelStrideWLimit.GetErrorOrEmpty(strideW);

    error += limits.kDilationLimit.GetErrorOrEmpty(dilationH, dilationW);
    return error;
}

bool Validator_35::ValidateCnn2D(const std::string& name,
                                 const uint32_t inHeight,
                                 const uint32_t inWidth,
                                 const uint32_t inChannels,
                                 const uint32_t kernelH,
                                 const uint32_t kernelW,
                                 const uint32_t kernelN,
                                 const uint32_t strideH,
                                 const uint32_t strideW,
                                 const uint32_t dilationH,
                                 const uint32_t dilationW,
                                 const OvGnaType inPrecision,
                                 const bool throwOnError) const {
    auto error = ValidateCnn(kCnn2DLimits,
                             name,
                             inHeight,
                             inWidth,
                             inChannels,
                             kernelH,
                             kernelW,
                             kernelN,
                             strideH,
                             strideW,
                             dilationH,
                             dilationW,
                             inPrecision);
    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

bool Validator_35::ValidateCnn1D(const std::string& name,
                                 const uint32_t inHeight,
                                 const uint32_t inWidth,
                                 const uint32_t inChannels,
                                 const uint32_t kernelH,
                                 const uint32_t kernelW,
                                 const uint32_t kernelN,
                                 const uint32_t strideH,
                                 const uint32_t strideW,
                                 const uint32_t dilationH,
                                 const uint32_t dilationW,
                                 const OvGnaType inPrecision,
                                 const bool throwOnError) const {
    auto error = ValidateCnn(kCnn1DLimits,
                             name,
                             inHeight,
                             inWidth,
                             inChannels,
                             kernelH,
                             kernelW,
                             kernelN,
                             strideH,
                             strideW,
                             dilationH,
                             dilationW,
                             inPrecision);
    return ValidationSuccesful(throwOnError, error, name, "Convolution1D");
}

std::string Validator_35::ValidatePooling(const CnnLimits& limits,
                                          const std::string& name,
                                          const uint32_t windowH,
                                          const uint32_t windowW,
                                          const uint32_t strideH,
                                          const uint32_t strideW) const {
    auto error = limits.kPoolingWindowHWLimit.GetErrorOrEmpty(windowH, windowW);
    error += limits.kPoolingStrideHWLimit.GetErrorOrEmpty(strideH, strideW);

    const RangeLimit poolingStrideHLimit{1, windowH, "pooling stride height (must be up to pooling window height)"};
    const RangeLimit poolingStrideWLimit{1, windowW, "pooling stride width (must be up to pooling window width)"};

    error += poolingStrideHLimit.GetErrorOrEmpty(strideH);
    error += poolingStrideWLimit.GetErrorOrEmpty(strideW);

    return error;
}

bool Validator_35::ValidatePooling2D(const std::string& name,
                                     const uint32_t windowH,
                                     const uint32_t windowW,
                                     const uint32_t strideH,
                                     const uint32_t strideW,
                                     const bool throwOnError) const {
    auto error = ValidatePooling(kCnn2DLimits, name, windowH, windowW, strideH, strideW);
    return ValidationSuccesful(throwOnError, error, name, "Pooling2D");
}

bool Validator_35::ValidateInputPadding(const std::string& name,
                                        const uint32_t pad_h_begin,
                                        const uint32_t pad_h_end,
                                        const uint32_t pad_w_begin,
                                        const uint32_t pad_w_end,
                                        const uint32_t kernel_h,
                                        const uint32_t kernel_w,
                                        const bool throwOnError) const {
    const IsEqualToLimit padding_h_symetric{pad_h_end,
                                            "convolution input padding along height axis (must be symmetric)"};
    const IsEqualToLimit padding_w_symetric{pad_w_end,
                                            "convolution input padding along width axis (must be symmetric)"};

    const IsLessThanLimit padding_h_limit{kernel_h,
                                          "convolution input padding height (must be less than kernel height)"};
    const IsLessThanLimit padding_w_limit{kernel_w, "convolution input padding width (must be less than kernel width)"};

    auto error = padding_h_symetric.GetErrorOrEmpty(pad_h_begin);
    error += padding_w_symetric.GetErrorOrEmpty(pad_w_begin);

    error += padding_h_limit.GetErrorOrEmpty(pad_h_begin);
    error += padding_w_limit.GetErrorOrEmpty(pad_w_begin);

    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

bool Validator_35::ShouldUseOnlyConv2DGnaIface() const {
    return true;
}

std::shared_ptr<AbstractValidator> AbstractValidator::Create(const DeviceVersion& target) {
    switch (target) {
    case DeviceVersion::GNA3_0:
    case DeviceVersion::GNA3_1:
        return std::make_shared<Validator_30>();
    case DeviceVersion::GNA3_5:
    case DeviceVersion::GNAEmbedded3_5:
    case DeviceVersion::GNA3_6:
    case DeviceVersion::GNA4_0:
        return std::make_shared<Validator_35>();
    default:
        return nullptr;
    }
}

void AbstractValidator::ThrowIfNotEmpty(const std::string& prefix, const std::string& error) {
    if (!error.empty()) {
        THROW_GNA_EXCEPTION << prefix << error;
    }
}

bool AbstractValidator::ValidationSuccesful(const bool throwOnError,
                                            const std::string& error,
                                            const std::string& operationName,
                                            const std::string& type) {
    if (throwOnError) {
        const std::string prefix = "Layer " + type + ": " + operationName + ":";
        ThrowIfNotEmpty(prefix, error);
    }

    return error.empty();
}

}  // namespace cnn2d

constexpr uint32_t Limitations::kBufferMaxSize;
constexpr uint32_t Limitations::kConvMinFiltersNum;
constexpr uint32_t Limitations::kConvMaxFiltersNum;
constexpr uint32_t Limitations::kConvDilationHeight;
constexpr uint32_t Limitations::kConvDilationWidth;
constexpr uint32_t Limitations::kConvFiltersNumDivider;
constexpr uint32_t Limitations::kConvFilterSizeDivider;
constexpr uint32_t Limitations::kConvFilterMaxSize;
constexpr uint32_t Limitations::kConvEachKernelByteAlignment;
constexpr uint32_t Limitations::kNoOfInputsDivisor;
constexpr uint32_t Limitations::kNoOfInputsLowPrecDivisor;
constexpr uint32_t Limitations::kAffineMaxBatchSize;
constexpr uint32_t Limitations::kMaxPoolMaxWindowSize;
constexpr uint32_t Limitations::kCopyMaxGrouping;
constexpr uint32_t Limitations::kTransposeMaxSize;
constexpr uint32_t Limitations::kMaxLayersCountGNA1_0;
constexpr uint32_t Limitations::kMaxLayersCountGNA2_0;
constexpr uint32_t Limitations::kMaxLayersCountGNA3_X;
constexpr uint32_t Limitations::kBytesPerSplitElement;
constexpr uint32_t Limitations::kBytesPerCropElement;
constexpr uint32_t Limitations::kBytesPerConcatElement;
constexpr uint32_t Limitations::kMemoryPageSize;

std::unordered_map<std::thread::id, std::shared_ptr<Limitations>> Limitations::kInstances;
std::mutex Limitations::kInstancesMtx;

Limitations::Limitations(const DeviceVersion& target) {
    m_use_only_16bit_conv_weights =
        (target == DeviceVersion::GNA1_0 || target == DeviceVersion::GNAEmbedded1_0 ||
         target == DeviceVersion::GNA2_0 || target == DeviceVersion::GNA3_0 || target == DeviceVersion::GNA3_1);

    m_mem_alignment = get_memory_alignment_bytes(target);
    m_cnn_validator = cnn2d::AbstractValidator::Create(target);
}

void Limitations::init(const DeviceVersion& compile_target) {
    std::lock_guard<std::mutex> lock(kInstancesMtx);
    auto thread_id = std::this_thread::get_id();
    kInstances[thread_id] = std::shared_ptr<Limitations>(new Limitations(compile_target));
}

void Limitations::deinit() {
    std::lock_guard<std::mutex> lock(kInstancesMtx);
    auto thread_id = std::this_thread::get_id();
    auto iter = kInstances.find(std::this_thread::get_id());
    if (iter != kInstances.end()) {
        kInstances.erase(thread_id);
    }
}

size_t Limitations::get_min_batch_to_fit_in_buffer(InferenceEngine::DataPtr input) {
    auto total_size = InferenceEngine::details::product(std::begin(input->getDims()), std::end(input->getDims()));
    return total_size / kBufferMaxSize + 1;
}

size_t Limitations::get_memory_alignment_bytes(const DeviceVersion& target) const {
    static const std::unordered_map<DeviceVersion, size_t> mem_alignment_map{{DeviceVersion::GNA1_0, 64},
                                                                             {DeviceVersion::GNAEmbedded1_0, 64},
                                                                             {DeviceVersion::GNA2_0, 64},
                                                                             {DeviceVersion::GNA3_0, 64},
                                                                             {DeviceVersion::GNA3_1, 64},
                                                                             {DeviceVersion::GNA3_5, 64},
                                                                             {DeviceVersion::GNAEmbedded3_5, 64},
                                                                             {DeviceVersion::GNA3_6, 16},
                                                                             {DeviceVersion::GNA4_0, 16}};

    return common::GetValueForKey<DeviceVersion, size_t>(target, mem_alignment_map);
}

bool SupportedElementTypes::IsParameterTypeSupported(ov::element::Type elem_type, bool is_exception_allowed) {
    if (supported_parameter_types.count(elem_type) == 0) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "The plugin does not support input precision with " << elem_type.get_type_name()
                                << " format. Supported precisions " << supported_parameter_types << "\n";
        }
        return false;
    }
    return true;
}

const std::set<ov::element::Type> SupportedElementTypes::supported_constant_types = {ov::element::i8,
                                                                                     ov::element::u8,
                                                                                     ov::element::i16,
                                                                                     ov::element::u16,
                                                                                     ov::element::i32,
                                                                                     ov::element::f32,
                                                                                     ov::element::f64};

bool SupportedElementTypes::IsConstantTypeSupported(ov::element::Type elem_type, bool is_exception_allowed) {
    if (supported_constant_types.count(elem_type) == 0) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "The plugin does not support constant precision with " << elem_type.get_type_name()
                                << " format. Supported precisions " << supported_constant_types << "\n";
        }
        return false;
    }
    return true;
}

bool Limitations::is_transpose_supported(const ov::Shape& shape) {
    const ov::Shape squeezed_shape = graph_utils::squeeze_shape(shape);

    // GNA transpose limitations:
    // - supports 2d transposes only
    // - smaller dimension should be less or equal to 8
    // - bigger dimension should be a multiple of limitations::noOfInputsDivisor
    if (squeezed_shape.size() == 2) {
        const size_t min_input_dim = std::min(squeezed_shape[0], squeezed_shape[1]);
        const size_t max_input_dim = std::max(squeezed_shape[0], squeezed_shape[1]);
        if (min_input_dim <= 8 && max_input_dim % Limitations::kNoOfInputsDivisor == 0 &&
            max_input_dim <= kTransposeMaxSize) {
            return true;
        }
    } else if (graph_utils::is_one_dim_shape(squeezed_shape)) {
        // it means that transpose input has only one dimension > 1
        return true;
    }
    return false;
}

bool Limitations::is_transpose_supported(const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "Transpose node is empty!");
    return is_transpose_supported(node->get_input_shape(0));
}

bool Limitations::is_conv_supported(const std::shared_ptr<ov::intel_gna::op::GNAConvolution>& conv_gna,
                                    const InferenceEngine::Precision gna_precision,
                                    bool is_exception_allowed) {
    OPENVINO_ASSERT(conv_gna, "GNAConvolution node is empty!");
    size_t batch_size = conv_gna->input_value(0).get_shape()[0];
    if (batch_size != 1) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "topology with layer: " + conv_gna->get_friendly_name() +
                                       ", type: " + conv_gna->get_type_name() + ", and batch size(" +
                                       std::to_string(batch_size) + ") != 1 not supported";
        }
        return false;
    }
    auto check_dilation = [&](size_t filter_dilation_height, size_t filter_stride_width) -> bool {
        cnn2d::RangeLimit2D dilation_limit{{kConvDilationHeight, kConvDilationHeight, "dilation height"},
                                           {kConvDilationWidth, kConvDilationWidth, "dilation width"}};
        std::string error = dilation_limit.GetErrorOrEmpty(static_cast<uint32_t>(filter_dilation_height),
                                                           static_cast<uint32_t>(filter_stride_width));
        return cnn2d::AbstractValidator::ValidationSuccesful(is_exception_allowed,
                                                             error,
                                                             conv_gna->get_friendly_name(),
                                                             conv_gna->get_type_name());
    };
    auto input_shape = conv_gna->input_value(0).get_shape();
    auto filter_shape = conv_gna->input_value(1).get_shape();
    if ((4 == filter_shape.size() && filter_shape[1] > 1 && filter_shape[2] > 1) ||
        (4 == input_shape.size() && input_shape[1] > 1 && input_shape[2] > 1)) {
        pass::helper::ConvData conv_data;
        pass::helper::GetConvData(conv_gna, conv_data);
        if (gna_convolution_layer::isMappableFrom2DTo1D(static_cast<uint32_t>(conv_data.input_height),
                                                        static_cast<uint32_t>(conv_data.input_width),
                                                        static_cast<uint32_t>(conv_data.input_channel_count),
                                                        static_cast<uint32_t>(conv_data.filter_height),
                                                        static_cast<uint32_t>(conv_data.filter_width),
                                                        static_cast<uint32_t>(conv_data.filter_stride_height),
                                                        static_cast<uint32_t>(conv_data.filter_stride_width))) {
            return check_dilation(conv_data.filter_dilation_height, conv_data.filter_dilation_width);
        }

        if (m_cnn_validator) {
            return m_cnn_validator->ValidateCnn2D(conv_gna->get_friendly_name(),
                                                  static_cast<uint32_t>(conv_data.input_height),
                                                  static_cast<uint32_t>(conv_data.input_width),
                                                  static_cast<uint32_t>(conv_data.input_channel_count),
                                                  static_cast<uint32_t>(conv_data.filter_height),
                                                  static_cast<uint32_t>(conv_data.filter_width),
                                                  static_cast<uint32_t>(conv_data.filter_channel_count),
                                                  static_cast<uint32_t>(conv_data.filter_stride_height),
                                                  static_cast<uint32_t>(conv_data.filter_stride_width),
                                                  static_cast<uint32_t>(conv_data.filter_dilation_height),
                                                  static_cast<uint32_t>(conv_data.filter_dilation_width),
                                                  OvGnaTypeIntFromBytes(gna_precision.size()),
                                                  is_exception_allowed);
        }
    }

    return check_dilation(conv_gna->get_dilations()[0],
                          conv_gna->get_dilations()[conv_gna->get_dilations().size() - 1]);
}

bool Limitations::is_pooling_supported(const std::shared_ptr<ov::intel_gna::op::GNAMaxPool> max_pool,
                                       bool is_exception_allowed) {
    OPENVINO_ASSERT(max_pool, "MaxPool node is empty!");
    auto kernels = max_pool->get_kernel();
    if (2 == kernels.size() && kernels[0] > 1 && kernels[1] > 1) {
        if (m_cnn_validator) {
            auto strides = max_pool->get_strides();
            return m_cnn_validator->ValidatePooling2D(max_pool->get_friendly_name(),
                                                      static_cast<uint32_t>(kernels[0]),
                                                      static_cast<uint32_t>(kernels[1]),
                                                      static_cast<uint32_t>(strides[0]),
                                                      static_cast<uint32_t>(strides[1]),
                                                      is_exception_allowed);
        }
    }
    return true;
}

bool Limitations::is_fc_supported(const std::shared_ptr<ngraph::op::FullyConnected>& fully_connected,
                                  bool is_exception_allowed) {
    OPENVINO_ASSERT(fully_connected, "FullyConnected node is empty!");
    size_t output_batch_size = fully_connected->get_output_shape(0)[0];
    if (output_batch_size > 8) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "topology with layer: " + fully_connected->get_friendly_name() +
                                       ", type: " + fully_connected->get_type_name() + ", and batch size(" +
                                       std::to_string(output_batch_size) + ") not supported";
        }
        return false;
    }
    return true;
}

bool Limitations::is_split_supported(const std::shared_ptr<ov::Node>& node, bool is_exception_allowed) {
    OPENVINO_ASSERT(node, "Split node is empty!");
    bool is_aligned = true;
    for (size_t i = 0; i < node->get_output_size(); i++) {
        is_aligned &= ov::intel_gna::graph_utils::is_aligned_split(node, i);
    }
    return is_aligned;
}

bool Limitations::is_concat_supported(const std::shared_ptr<const ov::Node>& node, bool is_exception_allowed) {
    OPENVINO_ASSERT(node, "Concat node is empty!");
    auto concat_node = std::dynamic_pointer_cast<const Concat>(node);
    const ov::Shape& concat_shape_out = concat_node->get_output_shape(0);
    auto axis = concat_node->get_axis();

    std::function<bool(std::shared_ptr<ov::Node>)> is_skipped_layer = [](std::shared_ptr<ov::Node> node) {
        return graph_utils::is_non_functional(node) || graph_utils::is_split(node) || graph_utils::is_copy(node) ||
               graph_utils::is_activation(node);
    };

    size_t skipped_ops_count = 0;
    bool is_interleaved = false;
    for (size_t i = 0; i < concat_node->inputs().size(); ++i) {
        auto concat_input =
            graph_utils::get_prev_node_skipping_certain(concat_node->get_input_node_shared_ptr(i), is_skipped_layer);
        if (ov::op::util::is_parameter(concat_input) || ov::op::util::is_constant(concat_input)) {
            skipped_ops_count++;
        }
        const ov::Shape concat_input_shape = concat_input->get_output_shape(0);
        // graph compiler changes the concat axis if one of the inputs is interleaved layer output
        if (graph_utils::squeeze_shape(concat_input_shape).size() >= 2 && graph_utils::is_interleaved(concat_input)) {
            is_interleaved = true;
        }
    }
    bool is_supported = false;
    if (skipped_ops_count == concat_node->inputs().size()) {
        is_supported = true;
    } else if (is_interleaved) {
        // TODO: need to extend interleaved layers detection patterns when migration to ngraph is finished.
        // make interleaved shape
        ov::Shape tr_shape(concat_shape_out);
        std::rotate(tr_shape.begin(), tr_shape.begin() + 1, tr_shape.end());

        // make interleaved order
        std::vector<size_t> tr_order(concat_shape_out.size());
        std::iota(tr_order.begin(), tr_order.end(), 0);
        std::rotate(tr_order.begin(), tr_order.begin() + 1, tr_order.end());

        const int64_t tr_axis = std::distance(tr_order.begin(), std::find(tr_order.begin(), tr_order.end(), axis));

        is_supported = graph_utils::get_first_valuable_dim_id(tr_shape) == tr_axis;
    } else {
        is_supported = graph_utils::get_first_valuable_dim_id(concat_shape_out) == axis;
    }

    if (!is_supported && is_exception_allowed) {
        THROW_GNA_EXCEPTION << concat_node->get_friendly_name()
                            << " Unsupported concatenation axis=" << concat_node->get_axis()
                            << " for input dimensions: " << concat_node->get_input_shape(0);
    }

    return is_supported;
}

bool Limitations::is_forward_transposed_concat_supported(const std::shared_ptr<const ov::Node>& node,
                                                         const AxisVector& order) {
    auto concat_node = std::dynamic_pointer_cast<const Concat>(node);
    if (!concat_node) {
        log::debug() << "Concat node is empty!" << std::endl;
        return false;
    }

    const ov::Shape& output_shape = concat_node->get_output_shape(0);
    auto axis = concat_node->get_axis();

    const ov::Shape& transposed_shape =
        graph_utils::transpose_shape(output_shape, pass::helper::reverse_transpose_order(order));
    const size_t transposed_concat_axis = order[axis];

    return graph_utils::get_first_valuable_dim_id(transposed_shape) == static_cast<int64_t>(transposed_concat_axis);
}

bool Limitations::is_backward_transposed_concat_supported(const std::shared_ptr<const ov::Node>& node,
                                                          const AxisVector& order) {
    auto concat_node = std::dynamic_pointer_cast<const Concat>(node);
    if (!concat_node) {
        log::debug() << "Concat node is empty!" << std::endl;
        return false;
    }

    const ov::Shape& output_shape = concat_node->get_output_shape(0);
    auto axis = concat_node->get_axis();

    const ov::Shape& transposed_shape = graph_utils::transpose_shape(output_shape, order);
    const size_t transposed_concat_axis = order[axis];

    return graph_utils::get_first_valuable_dim_id(transposed_shape) == static_cast<int64_t>(transposed_concat_axis);
}

bool Limitations::is_forward_transposed_split_supported(const std::shared_ptr<const ov::Node>& node,
                                                        const AxisVector& order) {
    std::shared_ptr<const ov::Node> split_node = nullptr;
    if (std::dynamic_pointer_cast<const Split>(node)) {
        split_node = std::dynamic_pointer_cast<const Split>(node);
    } else if (std::dynamic_pointer_cast<const VariadicSplit>(node)) {
        split_node = std::dynamic_pointer_cast<const VariadicSplit>(node);
    } else {
        log::debug() << "Split node is empty!" << std::endl;
        return false;
    }

    const ov::Shape& output_shape = split_node->get_output_shape(0);
    auto constant_node = as_type_ptr<Constant>(split_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return false;
    auto axis = constant_node->get_axis_vector_val()[0];

    const ov::Shape& transposed_shape =
        graph_utils::transpose_shape(output_shape, pass::helper::reverse_transpose_order(order));
    const size_t transposed_concat_axis = order[axis];

    return graph_utils::get_first_valuable_dim_id(transposed_shape) == static_cast<int64_t>(transposed_concat_axis);
}

bool Limitations::is_backward_transposed_split_supported(const std::shared_ptr<const ov::Node>& node,
                                                         const AxisVector& order) {
    std::shared_ptr<const ov::Node> split_node = nullptr;
    if (std::dynamic_pointer_cast<const Split>(node)) {
        split_node = std::dynamic_pointer_cast<const Split>(node);
    } else if (std::dynamic_pointer_cast<const VariadicSplit>(node)) {
        split_node = std::dynamic_pointer_cast<const VariadicSplit>(node);
    } else {
        log::debug() << "Split node is empty!" << std::endl;
        return false;
    }

    const ov::Shape& output_shape = split_node->get_output_shape(0);
    auto constant_node = as_type_ptr<Constant>(split_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return false;
    auto axis = constant_node->get_axis_vector_val()[0];

    const ov::Shape& transposed_shape =
        graph_utils::transpose_shape(output_shape, pass::helper::reverse_transpose_order(order));
    const int64_t transposed_concat_axis = order[axis];

    return graph_utils::get_first_valuable_dim_id(transposed_shape) == transposed_concat_axis;
}

bool Limitations::is_op_supported(const std::shared_ptr<ov::Node>& node,
                                  const InferenceEngine::Precision gna_precision,
                                  bool is_exception_allowed) {
    if (ov::op::util::is_parameter(node)) {
        return SupportedElementTypes::IsParameterTypeSupported(node->get_element_type(), is_exception_allowed);
    } else if (ov::op::util::is_constant(node)) {
        return SupportedElementTypes::IsConstantTypeSupported(node->get_element_type(), is_exception_allowed);
    } else if (auto conv = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(node)) {
        return is_conv_supported(conv, gna_precision, is_exception_allowed);
    } else if (auto concat = std::dynamic_pointer_cast<Concat>(node)) {
        return is_concat_supported(concat, is_exception_allowed);
    } else if (auto fully_connected = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node)) {
        return is_fc_supported(fully_connected, is_exception_allowed);
    } else if (ov::intel_gna::graph_utils::is_pooling(node)) {
        return is_pooling_supported(std::dynamic_pointer_cast<ov::intel_gna::op::GNAMaxPool>(node),
                                    is_exception_allowed);
    } else if (ov::op::util::is_output(node) || ov::op::util::is_sink(node) ||
               ov::intel_gna::graph_utils::is_eltwise_add(node) || ov::intel_gna::graph_utils::is_eltwise_mul(node) ||
               ov::intel_gna::graph_utils::is_crop_affined(node) ||
               ov::intel_gna::graph_utils::is_activation(node.get()) ||
               ov::intel_gna::graph_utils::is_gna_precision_agnostic(
                   node) ||  // check concat/split are aligned when transformations will be moved to ngraph
               (std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(node) != nullptr) ||
               (std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(node) != nullptr) ||
               (std::dynamic_pointer_cast<ngraph::op::PowerIE>(node) != nullptr) ||
               (std::dynamic_pointer_cast<MatMul>(node) != nullptr)) {
        return true;
    } else if (ov::intel_gna::graph_utils::is_gna_precision_agnostic(node)) {
        if ((std::dynamic_pointer_cast<Split>(node) != nullptr) ||
            (std::dynamic_pointer_cast<VariadicSplit>(node) != nullptr)) {
            return is_split_supported(node, is_exception_allowed);
        }
        // TODO check concat are aligned when transformation will be moved to ngraph
        return true;
    }
    return false;
}

void Limitations::check_all_ops_supported(const std::shared_ptr<ov::Model>& model,
                                          const InferenceEngine::Precision gna_precision) {
    std::stringstream error;
    // Walk through the transformed model
    for (auto& op : model->get_ops()) {
        try {
            if (!is_op_supported(op, gna_precision, true)) {
                error << "The plugin does not support layer " << op->get_friendly_name() << " (type "
                      << op->get_type_name() << ")!" << std::endl;
            }
        } catch (const InferenceEngine::GeneralError& e) {
            error << e.what() << std::endl;
        }
    }
    if (!error.str().empty()) {
        THROW_GNA_EXCEPTION << error.str();
    }
}

bool Limitations::use_only_16bit_convolution_weights() const {
    return m_use_only_16bit_conv_weights;
}

bool Limitations::validate_conv_concat_axis(const InferenceEngine::ConcatLayer* concat_layer) {
    IE_ASSERT(concat_layer);
    auto dims_size = concat_layer->insData[0].lock()->getDims().size();

    if (dims_size >= 2) {
        InferenceEngine::CNNLayerPtr prev_layer;

        // Skipping here all layers which would disappear or otherwise fuse with convolution in the final GNA graph
        auto isFusableWithConv = [](InferenceEngine::CNNLayerPtr ptr) {
            return (LayerInfo(ptr).isFusableWithConv() || LayerInfo(ptr).isNonFunctional() ||
                    LayerInfo(ptr).isConcat());
        };

        auto in_dims = concat_layer->insData[0].lock()->getDims();
        auto concat_axis = concat_layer->_axis;
        auto concat_layout = concat_layer->input()->getLayout();

        for (size_t input_idx = 0; input_idx != concat_layer->insData.size(); input_idx++) {
            // Supported cases for concatenation of a convolution
            prev_layer = InferenceEngine::CNNNetPrevLayerSkipCertain(concat_layer,
                                                                     static_cast<int>(input_idx),
                                                                     isFusableWithConv);
            if (prev_layer && LayerInfo(prev_layer).isConvolution()) {
                // Allow concatenation along N axis for non-interleaved primitives
                // (currently only convolution)
                if (concat_layer->_axis == 0)
                    break;

                // Convert dims to NHWC layout to allow later verification
                auto new_order = permute::GetPermuteOrder(concat_layout, InferenceEngine::Layout::NHWC);
                InferenceEngine::SizeVector new_dims;
                for (size_t i = 0; i < dims_size; ++i) {
                    new_dims.push_back(in_dims[new_order[i]]);
                }
                concat_axis = permute::GetPermuteOrder(InferenceEngine::Layout::NHWC, concat_layout)[concat_axis];

                // Looking for any axis with dimension > 1 before concatentaion axis;
                // in general such concatenation is unsupported
                auto end_dim = new_dims.begin() + concat_axis;
                auto unsupportedconcat_axis = std::find_if(new_dims.begin(), end_dim, [](const size_t& inDim) {
                    return (inDim > 1);
                });

                if (unsupportedconcat_axis != end_dim) {
                    return false;
                }

                break;
            }
        }
    }
    return true;
}

bool Limitations::are_layers_supported(InferenceEngine::CNNNetwork& network, std::string& errMessage) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::unordered_set<InferenceEngine::CNNLayer*> allLayers;
    InferenceEngine::CNNLayerPtr startLayer;
    if (inputs.empty()) {
        auto outputs = network.getOutputsInfo();
        IE_ASSERT(!outputs.empty());
        // If there are no inputs start search from an output
        startLayer = getCreatorLayer(outputs.begin()->second).lock();
    } else {
        SupportedElementTypes::IsParameterTypeSupported(
            InferenceEngine::details::convertPrecision(inputs.begin()->second->getPrecision()),
            true);

        auto& secondLayers = getInputTo(inputs.begin()->second->getInputData());
        if (secondLayers.empty()) {
            errMessage = "Network consists of input layer only (GNA)\n";
            return false;
        }
        startLayer = secondLayers.begin()->second;
    }
    auto batch_size = network.getBatchSize();
    bool check_result = true;
    InferenceEngine::details::UnorderedDFS(
        allLayers,
        startLayer,
        [&](const InferenceEngine::CNNLayerPtr layer) {
            LayerInfo info(layer);
            if (LayerTypeFromStr(layer->type) == LayerType::NO_TYPE) {
                errMessage = "The plugin does not support layer: " + layer->name + ":" + layer->type + "\n";
                check_result = false;
            }
            if (batch_size != 1 && info.isBatchSizeConstrained()) {
                errMessage = "topology with layer: " + layer->name + ", type: " + layer->type + ", and batch size(" +
                             std::to_string(batch_size) + ") != 1 not supported";
                check_result = false;
            }
            if (info.isFullyConnected()) {
                size_t output_batch_size = info.getOutputBatchSize();
                if (output_batch_size > 8) {
                    errMessage = "topology with layer: " + layer->name + ", type: " + layer->type +
                                 ", and batch size(" + std::to_string(output_batch_size) + ") not supported";
                    check_result = false;
                }
            }
        },
        false);
    return check_result;
}
IE_SUPPRESS_DEPRECATED_END

}  // namespace limitations
}  // namespace intel_gna
}  // namespace ov
