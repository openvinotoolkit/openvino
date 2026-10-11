// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief Fuses temporal begin padding, anisotropic space-to-depth, and grouped mean reduction.
class GroupedSpaceToDepth : public ov::op::Op {
public:
    OPENVINO_OP("GroupedSpaceToDepth", "ie_internal_opset");

    GroupedSpaceToDepth() = default;
    GroupedSpaceToDepth(const ov::Output<Node>& input, size_t factor_t, size_t factor_s, size_t output_channels);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    size_t get_factor_t() const {
        return m_factor_t;
    }
    size_t get_factor_s() const {
        return m_factor_s;
    }
    size_t get_output_channels() const {
        return m_output_channels;
    }

private:
    size_t m_factor_t = 1;
    size_t m_factor_s = 1;
    size_t m_output_channels = 0;
};

}  // namespace ov::intel_gpu::op
