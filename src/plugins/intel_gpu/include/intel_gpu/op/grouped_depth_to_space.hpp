// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief Rearranges repeated channel groups of a 5D tensor into temporal and spatial dimensions.
///
/// The input has shape [N, C, T, H, W]. Channels are repeated as needed and interpreted as
/// [output_channels, factor_t, factor_s, factor_s] before being moved into the T, H, and W dimensions.
/// crop_begin_t elements are removed from the beginning of the expanded temporal dimension.
class GroupedDepthToSpace : public ov::op::Op {
public:
    OPENVINO_OP("GroupedDepthToSpace", "ie_internal_opset");

    GroupedDepthToSpace() = default;
    GroupedDepthToSpace(const ov::Output<Node>& input, size_t factor_t, size_t factor_s, size_t output_channels, size_t crop_begin_t);

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
    size_t get_crop_begin_t() const {
        return m_crop_begin_t;
    }

private:
    size_t m_factor_t = 1;
    size_t m_factor_s = 1;
    size_t m_output_channels = 0;
    size_t m_crop_begin_t = 0;
};

}  // namespace ov::intel_gpu::op
