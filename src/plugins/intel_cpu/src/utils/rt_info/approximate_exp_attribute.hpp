// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov::intel_cpu {

/**
 * @brief Marks an Exp whose value is divided by a sum of itself along the reduction axis, i.e. the
 * numerator of a softmax. Set by ov::intel_cpu::pass::MarkApproximateSoftmaxExp and read by
 * jit_exp_emitter, which then evaluates the exponential with a degree-1 polynomial instead of the
 * accurate one. The decision is taken in a transformation rather than in the emitter so that it is
 * made on the graph shape SoftmaxDecomposition produces, before later passes can obscure it.
 */
class ApproximateExp : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("approximate_exp");
    ApproximateExp() = default;
    ~ApproximateExp() override;

    // The mark states something about this node's own consumers, so it must never be carried onto
    // another node by a pass that copies run-time info.
    [[nodiscard]] bool is_copyable() const override {
        return false;
    }
};

void mark_as_approximate_exp(const std::shared_ptr<ov::Node>& node);
bool is_approximate_exp(const std::shared_ptr<const ov::Node>& node);

}  // namespace ov::intel_cpu
