// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <legacy/ie_layers.h>

#include <cstdint>
#include <ie_algorithm.hpp>
#include <memory>
#include <thread>

#include "common/gna_target.hpp"
#include "common/misc_utils.hpp"
#include "dnn_types.hpp"
#include "gna_lib_ver_selector.hpp"
#include "legacy/ngraph_ops/convolution_ie.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "ops/gna_convolution.hpp"
#include "ops/gna_max_pool.hpp"

namespace ov {
namespace intel_gna {
namespace limitations {

namespace cnn2d {

struct IsEqualToLimit {
    uint32_t compared_value;
    std::string what;
    bool IsValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct IsLessThanLimit {
    uint32_t compared_value;
    std::string what;
    bool IsValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct RangeLimit {
    uint32_t min;
    uint32_t max;
    std::string what;
    bool IsValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct RangeLimit2D {
    RangeLimit hLimit;
    RangeLimit wLimit;
    bool IsValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w) const;
};

struct RangeMultipleLimit : public RangeLimit {
    uint32_t multiplier;
    RangeMultipleLimit(RangeLimit rlIn, uint32_t multiplierIn);
    bool IsValid(const uint32_t val) const;
    std::string GetErrorOrEmpty(const uint32_t val) const;
};

struct RectLimit {
    uint32_t maxVectorHeight;
    uint32_t maxVectorWidth;
    bool IsValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const;
};

struct VectorOrSquareLimit {
    uint32_t maxSquare;
    uint32_t maxVectorHeight;
    uint32_t maxVectorWidth;
    bool IsValid(const uint32_t h, const uint32_t w) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const;
};

struct RectLimitByChannels {
    std::vector<std::pair<uint32_t, RectLimit>> limitPerChannel;
    RectLimit GetByChannels(const uint32_t channels) const;
    bool IsValid(const uint32_t h, const uint32_t w, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h, const uint32_t w, const uint32_t channels, std::string what) const;
};

struct RectLimitByChannelsAndPrecision {
    RectLimitByChannels limit_for_int8;
    RectLimitByChannels limit_for_int16;
    RectLimitByChannels GetByPrecision(const OvGnaType precision) const;
    bool IsValid(const uint32_t h, const uint32_t w, const OvGnaType precision, const uint32_t channels) const;
    std::string GetErrorOrEmpty(const uint32_t h,
                                const uint32_t w,
                                const OvGnaType precision,
                                const uint32_t channels,
                                std::string what) const;
};

class AbstractValidator {
protected:
    static void ThrowIfNotEmpty(const std::string& prefix, const std::string& error);

public:
    static bool ValidationSuccesful(const bool throwOnError,
                                    const std::string& error,
                                    const std::string& operation,
                                    const std::string& type);

    virtual ~AbstractValidator() = default;
    virtual bool ValidateCnn2D(const std::string& name,
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
                               bool exception = true) const = 0;

    virtual bool ValidatePooling2D(const std::string& name,
                                   const uint32_t windowH,
                                   const uint32_t windowW,
                                   const uint32_t strideH,
                                   const uint32_t strideW,
                                   bool exception = true) const = 0;

    virtual bool ValidateInputPadding(const std::string& name,
                                      const uint32_t pad_h_begin,
                                      const uint32_t pad_h_end,
                                      const uint32_t pad_w_begin,
                                      const uint32_t pad_w_end,
                                      const uint32_t kernel_h,
                                      const uint32_t kernel_w,
                                      const bool throwOnError = true) const = 0;

    virtual bool ShouldUseOnlyConv2DGnaIface() const = 0;

    virtual bool ValidateCnn1D(const std::string& name,
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
                               bool exception = true) const = 0;

    static std::shared_ptr<AbstractValidator> Create(const target::DeviceVersion& target);
};

}  // namespace cnn2d

class Limitations {
public:
    /**
     * @brief Create an instance of the Limitations class. Since Limitations is designed as a singleton, multiple
     * instances of the plugin with different compilation targets cannot coexist simultaneously for the same thread.
     * @param compile_target GNA compile target
     */
    static void init(const target::DeviceVersion& compile_target);

    /**
     * @brief Delete the instance of the Limitations class for the currently running thread.
     */
    static void deinit();

    /**
     * @brief Returns the instance of Limitations object. Requires an Init call before the first usage
     */
    static inline std::shared_ptr<Limitations> get_instance();

    static size_t get_min_batch_to_fit_in_buffer(InferenceEngine::DataPtr input);

    /**
     * @brief Validates if concat layer axis is supported by GNA
     * @param layer concat layer
     * @return true if concat layer axis is valid
     */
    IE_SUPPRESS_DEPRECATED_START
    static bool validate_conv_concat_axis(const InferenceEngine::ConcatLayer* concatLayer);
    static bool are_layers_supported(InferenceEngine::CNNNetwork& network, std::string& errMessage);
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Validates if fully connected is supported by GNA
     * @param fully_connected fully connected
     * @param is_exception_allowed flag specifies whether exception is allowed
     * @return true if supported
     */
    static bool is_fc_supported(const std::shared_ptr<ngraph::op::FullyConnected>& fully_connected,
                                bool is_exception_allowed = false);
    /**
     * @brief Validates if split is supported by GNA
     * @param node split
     * @param is_exception_allowed flag specifies whether exception is allowed
     * @return true if supported
     */
    static bool is_split_supported(const std::shared_ptr<ov::Node>& node, bool is_exception_allowed = false);

    /**
     * @brief Validates if transpose is supported by GNA
     * @param shape transpose
     * @return true if supported
     */
    static bool is_transpose_supported(const ov::Shape& shape);
    /**
     * @brief Validates if transpose is supported by GNA
     * @param node transpose
     * @return true if supported
     */
    static bool is_transpose_supported(const std::shared_ptr<const ov::Node>& node);
    /**
     * @brief Validates if convolution is supported by GNA
     * @param conv_gna GNA convolution
     * @param gna_precision GNA inference precision
     * @param is_exception_allowed flag specifies whether exception is allowed
     * @return true if supported
     */
    bool is_conv_supported(const std::shared_ptr<ov::intel_gna::op::GNAConvolution>& conv_gna,
                           const InferenceEngine::Precision gna_precision,
                           bool is_exception_allowed = false);
    /**
     * @brief Validates if max pooling is supported by GNA
     * @param max_pool max pooling
     * @param is_exception_allowed flag specifies whether exception is allowed
     * @return true if precision is found in supported
     */
    bool is_pooling_supported(const std::shared_ptr<ov::intel_gna::op::GNAMaxPool> max_pool,
                              bool is_exception_allowed = false);

    static bool is_concat_supported(const std::shared_ptr<const ov::Node>& node, bool is_exception_allowed);
    static bool is_forward_transposed_concat_supported(const std::shared_ptr<const ov::Node>& node,
                                                       const AxisVector& order);
    static bool is_backward_transposed_concat_supported(const std::shared_ptr<const ov::Node>& node,
                                                        const AxisVector& order);
    static bool is_forward_transposed_split_supported(const std::shared_ptr<const ov::Node>& node,
                                                      const AxisVector& order);
    static bool is_backward_transposed_split_supported(const std::shared_ptr<const ov::Node>& node,
                                                       const AxisVector& order);

    /**
     * @brief Validates if operation is supported by GNA
     * @param node operation
     * @param gna_precision GNA inference precision
     * @param is_exception_allowed flag specifies whether exception is allowed
     * @return true if supported
     */
    bool is_op_supported(const std::shared_ptr<ov::Node>& node,
                         const InferenceEngine::Precision gna_precision,
                         bool is_exception_allowed = false);

    /**
     * @brief Check if all operations are supported by GNA
     * @param model ngraph model
     * @param gna_precision GNA inference precision
     */
    void check_all_ops_supported(const std::shared_ptr<ov::Model>& model,
                                 const InferenceEngine::Precision gna_precision);

    bool use_only_16bit_convolution_weights() const;
    bool is_crop_affined_offset(size_t numberOfElements) const;
    bool is_aligned(size_t addr) const;
    size_t get_memory_alignment() const;
    std::shared_ptr<cnn2d::AbstractValidator> get_cnn_validator() const;

    constexpr static uint32_t kBufferMaxSize = 65528;
    constexpr static uint32_t kConvMinFiltersNum = 4;
    constexpr static uint32_t kConvMaxFiltersNum = 65532;
    constexpr static uint32_t kConvDilationHeight = 1;
    constexpr static uint32_t kConvDilationWidth = 1;
    constexpr static uint32_t kConvFiltersNumDivider = 4;
    constexpr static uint32_t kConvFilterSizeDivider = 8;
    constexpr static uint32_t kConvFilterMaxSize = 768;
    constexpr static uint32_t kConvEachKernelByteAlignment = 16;
    constexpr static uint32_t kNoOfInputsDivisor = 8;
    constexpr static uint32_t kNoOfInputsLowPrecDivisor = 16;
    constexpr static uint32_t kAffineMaxBatchSize = 8;
    constexpr static uint32_t kMaxPoolMaxWindowSize = 6;
    constexpr static uint32_t kCopyMaxGrouping = 8;
    constexpr static uint32_t kTransposeMaxSize = 65528;
    constexpr static uint32_t kMaxLayersCountGNA1_0 = 1023;
    constexpr static uint32_t kMaxLayersCountGNA2_0 = 4096;
    constexpr static uint32_t kMaxLayersCountGNA3_X = 8192;

    // Currently split layer only supports 2 bytes in int16 and int8 mode.
    // In fp32 mode this is not necessary but is useful for testing
    constexpr static uint32_t kBytesPerSplitElement = 2;
    // Currently crop layer only supports 2 bytes in int16 and int8 mode.
    // In fp32 mode this is not necessary but is useful for testing
    constexpr static uint32_t kBytesPerCropElement = 2;
    // currently concat layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this no necessary but usefull
    // for testing
    constexpr static uint32_t kBytesPerConcatElement = 2;
    constexpr static uint32_t kMemoryPageSize = 4096;

private:
    Limitations(const target::DeviceVersion& target);
    Limitations(const Limitations&) = delete;
    Limitations& operator=(const Limitations&) = delete;

    size_t get_memory_alignment_bytes(const target::DeviceVersion& target) const;

    bool m_use_only_16bit_conv_weights = false;
    size_t m_mem_alignment = 0;
    std::shared_ptr<cnn2d::AbstractValidator> m_cnn_validator;

    static std::unordered_map<std::thread::id, std::shared_ptr<Limitations>> kInstances;
    static std::mutex kInstancesMtx;
};

inline std::shared_ptr<Limitations> Limitations::get_instance() {
    std::lock_guard<std::mutex> lock(kInstancesMtx);
    auto thread_id = std::this_thread::get_id();
    auto iter = kInstances.find(thread_id);
    if (iter == kInstances.end() || !iter->second) {
        THROW_GNA_EXCEPTION << "Limitations instance is not initialized.\n";
    }
    return iter->second;
}

inline bool Limitations::is_crop_affined_offset(size_t numberOfElements) const {
    const auto cropOffset = numberOfElements * kBytesPerCropElement;
    return !is_aligned(cropOffset);
}

inline bool Limitations::is_aligned(size_t addr) const {
    return (addr == ALIGN(addr, get_memory_alignment()));
}

inline size_t Limitations::get_memory_alignment() const {
    return m_mem_alignment;
}

inline std::shared_ptr<cnn2d::AbstractValidator> Limitations::get_cnn_validator() const {
    return m_cnn_validator;
}

}  // namespace limitations
}  // namespace intel_gna
}  // namespace ov
