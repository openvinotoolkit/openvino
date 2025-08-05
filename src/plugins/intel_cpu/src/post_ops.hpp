// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using PostOps = std::vector<std::any>;

struct ActivationPostOp;
using eltwiseExecutorCreatingStrategy = std::function<ExecutorPtr(const ActivationPostOp&,
                                                                  ExecutorContext::CPtr,
                                                                  std::vector<MemoryDescPtr>,
                                                                  std::vector<MemoryDescPtr>,
                                                                  const PostOps&)>;

struct ActivationPostOp {
    enum class Type : uint8_t {
        relu,
        tanh,
        elu,
        square,
        abs,
        sqrt,
        soft_relu,
        logistic,
        exp,
        gelu_erf,
        gelu_tanh,
        clip,
        swish,
        hardswish,
        mish,
        hsigmoid,
        round_half_to_even,
        round_half_away_from_zero,
        linear,
        powerstatic,
        floor,
        negative,
        ceiling,
        erf,
        soft_sign,
        log,
    };

    ActivationPostOp(const Type type,
                     const float alpha,
                     const float beta,
                     const float gamma,
                     [[maybe_unused]] const eltwiseExecutorCreatingStrategy& strategy = nullptr)
        : m_type(type),
          m_alpha(alpha),
          m_beta(beta),
          m_gamma(gamma) {}

    [[nodiscard]] float alpha() const {
        return m_alpha;
    }

    [[nodiscard]] float beta() const {
        return m_beta;
    }

    [[nodiscard]] float gamma() const {
        return m_gamma;
    }

    [[nodiscard]] Type type() const {
        return m_type;
    }

private:
    const Type m_type;
    const float m_alpha;
    const float m_beta;
    const float m_gamma;
};

struct ScaleShiftPostOp {
    enum class Type : uint8_t {
        add,
        subtract,
        divide,
        multiply,
        muladd,
        power_dynamic,
        prelu,
        select,
        maximum,
        minimum,
        squared_difference,
        logical_and,
        logical_or,
        logical_xor,
        logical_not,
        floor_mod,
        mod,
        equal,
        not_equal,
        greater,
        greater_equal,
        less,
        less_equal,
        is_finite,
        is_inf,
        is_nan,
        bitwise_and,
        bitwise_not,
        bitwise_or,
        bitwise_xor
    };

    ScaleShiftPostOp(const Type m_type, std::vector<float> _scales, std::vector<float> _shifts)
        : m_type(m_type),
          m_scales(std::move(_scales)),
          m_shifts(std::move(_shifts)) {}

    [[nodiscard]] const std::vector<float>& scales() const {
        return m_scales;
    }

    [[nodiscard]] const std::vector<float>& shifts() const {
        return m_shifts;
    }

    [[nodiscard]] Type type() const {
        return m_type;
    }

private:
    const Type m_type;
    const std::vector<float> m_scales;
    const std::vector<float> m_shifts;
};

struct FakeQuantizePostOp {
    enum class Type : uint8_t { binarization, quantization_only, quantization_dequantization };

    FakeQuantizePostOp(const Type type,
                       std::vector<float> cropLow,
                       std::vector<float> cropHigh,
                       std::vector<float> inputScale,
                       std::vector<float> inputShift,
                       std::vector<float> outputScale,
                       std::vector<float> outputShift,
                       const size_t levels,
                       bool isInputLowBroadcasted,
                       bool isOutputHighBroadcasted)
        : m_type(type),
          m_cropLow(std::move(cropLow)),
          m_cropHigh(std::move(cropHigh)),
          m_inputScale(std::move(inputScale)),
          m_inputShift(std::move(inputShift)),
          m_outputScale(std::move(outputScale)),
          m_outputShift(std::move(outputShift)),
          m_levels(levels),
          m_isInputLowBroadcasted(isInputLowBroadcasted),
          m_isOutputHighBroadcasted(isOutputHighBroadcasted) {}

    [[nodiscard]] const std::vector<float>& cropLow() const {
        return m_cropLow;
    }

    [[nodiscard]] const std::vector<float>& cropHigh() const {
        return m_cropHigh;
    }

    [[nodiscard]] const std::vector<float>& inputScale() const {
        return m_inputScale;
    }

    [[nodiscard]] const std::vector<float>& inputShift() const {
        return m_inputShift;
    }

    [[nodiscard]] const std::vector<float>& outputScale() const {
        return m_outputScale;
    }

    [[nodiscard]] const std::vector<float>& outputShift() const {
        return m_outputShift;
    }

    [[nodiscard]] size_t levels() const {
        return m_levels;
    }

    [[nodiscard]] Type type() const {
        return m_type;
    }

    [[nodiscard]] bool isInputLowBroadcast() const {
        return m_isInputLowBroadcasted;
    }

    [[nodiscard]] bool isOutputHighBroadcast() const {
        return m_isOutputHighBroadcasted;
    }

private:
    const Type m_type;
    const std::vector<float> m_cropLow;
    const std::vector<float> m_cropHigh;
    const std::vector<float> m_inputScale;
    const std::vector<float> m_inputShift;
    const std::vector<float> m_outputScale;
    const std::vector<float> m_outputShift;
    const size_t m_levels;
    // necessary only for legacy post ops
    bool m_isInputLowBroadcasted;
    bool m_isOutputHighBroadcasted;
};

struct DepthwiseConvolutionPostOp {
    DepthwiseConvolutionPostOp(size_t ih, size_t iw, std::vector<size_t> kernel, std::vector<size_t> strides)
        : m_ih(ih),
          m_iw(iw),
          m_kernel(std::move(kernel)),
          m_strides(std::move(strides)) {}

    [[nodiscard]] size_t ih() const {
        return m_ih;
    }

    [[nodiscard]] size_t iw() const {
        return m_iw;
    }

    [[nodiscard]] const std::vector<size_t>& kernel() const {
        return m_kernel;
    }

    [[nodiscard]] const std::vector<size_t>& strides() const {
        return m_strides;
    }

private:
    size_t m_ih;
    size_t m_iw;
    std::vector<size_t> m_kernel;
    std::vector<size_t> m_strides;
};

struct SumPostOp {
    SumPostOp(float scale, int32_t zero_point, ov::element::Type_t dataType)
        : m_scale(scale),
          m_zero_point(zero_point),
          m_dataType(dataType) {}

    [[nodiscard]] float scale() const {
        return m_scale;
    }

    [[nodiscard]] int32_t zeroPoint() const {
        return m_zero_point;
    }

    [[nodiscard]] ov::element::Type_t dataType() const {
        return m_dataType;
    }

private:
    float m_scale;
    int32_t m_zero_point;
    ov::element::Type_t m_dataType;
};

enum class EltwiseKind : uint8_t {
    Activation,
    ScaleShift,
    // @todo Binary?
};

EltwiseKind getEltwiseKind(Algorithm alg);

ScaleShiftPostOp::Type convertToScaleShiftOpt(Algorithm alg);

ActivationPostOp::Type convertToActivationPostOpt(Algorithm alg);

Algorithm convertToEltwiseAlgorithm(ActivationPostOp::Type m_type);
Algorithm convertToEltwiseAlgorithm(ScaleShiftPostOp::Type type);
dnnl::algorithm convertToDnnlAlgorithm(ActivationPostOp::Type m_type);

FakeQuantizePostOp::Type convertToFqPostOp(Algorithm alg);

PostOps getPostOps(const std::vector<NodePtr>& fused, ov::element::Type_t sumDataType = ov::element::dynamic);
}  // namespace ov::intel_cpu
