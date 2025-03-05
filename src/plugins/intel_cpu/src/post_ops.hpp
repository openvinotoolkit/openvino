// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "node.h"
#include "nodes/executors/executor.hpp"

namespace ov::intel_cpu {

struct PostOp;
using PostOps = std::vector<std::shared_ptr<PostOp>>;

struct PostOp {
    virtual ~PostOp() = default;
};

struct ActivationPostOp;
using eltwiseExecutorCreatingStrategy = std::function<ExecutorPtr(const ActivationPostOp&,
                                                                  ExecutorContext::CPtr,
                                                                  std::vector<MemoryDescPtr>,
                                                                  std::vector<MemoryDescPtr>,
                                                                  const PostOps&)>;

struct ActivationPostOp : PostOp {
    enum class Type : size_t {
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
    };

    ActivationPostOp(const Type type,
                     const float alpha,
                     const float beta,
                     const float gamma,
                     const eltwiseExecutorCreatingStrategy& strategy = nullptr)
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

struct ScaleShiftPostOp : PostOp {
    enum Type {
        add,
        subtract,
        divide,
        multiply,
        muladd,
        powerstatic,
        prelu,
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

struct FakeQuantizePostOp : PostOp {
    FakeQuantizePostOp(std::vector<float> cropLow,
                       std::vector<float> cropHigh,
                       std::vector<float> inputScale,
                       std::vector<float> inputShift,
                       std::vector<float> outputScale,
                       std::vector<float> outputShift,
                       const size_t levels)
        : m_cropLow(std::move(cropLow)),
          m_cropHigh(std::move(cropHigh)),
          m_inputScale(std::move(inputScale)),
          m_inputShift(std::move(inputShift)),
          m_outputScale(std::move(outputScale)),
          m_outputShift(std::move(outputShift)),
          m_levels(levels) {}

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

private:
    const std::vector<float> m_cropLow;
    const std::vector<float> m_cropHigh;
    const std::vector<float> m_inputScale;
    const std::vector<float> m_inputShift;
    const std::vector<float> m_outputScale;
    const std::vector<float> m_outputShift;
    const size_t m_levels;
};

enum class EltwiseKind {
    Activation,
    ScaleShift,
    // @todo Binary?
};

using PostOps = std::vector<std::shared_ptr<PostOp>>;

EltwiseKind getEltwiseKind(const Algorithm alg);

ScaleShiftPostOp::Type convertToScaleShiftOpt(const Algorithm alg);

ActivationPostOp::Type convertToActivationPostOpt(const Algorithm alg);

Algorithm convertToEltwiseAlgorithm(const ActivationPostOp::Type m_type);

PostOps getPostOps(const std::vector<NodePtr>& fused);
}  // namespace ov::intel_cpu
