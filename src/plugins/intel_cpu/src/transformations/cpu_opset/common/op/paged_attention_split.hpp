// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"

namespace ov {
namespace intel_cpu {
/// \brief PagedAttention, fused with VariadicSplit/Reshape
///
/// \ingroup ov_ops_cpp_api

class PagedAttentionWithSplit : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionWithSplit", "cpu_plugin_opset");

    PagedAttentionWithSplit() = default;

    PagedAttentionWithSplit(const OutputVector& args, const Extensions::Cpu::PagedAttentionFuseConfig& cfg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const Extensions::Cpu::PagedAttentionFuseConfig& get_config() const {
        return m_config;
    }

    Extensions::Cpu::PagedAttentionFuseConfig& get_config() {
        return m_config;
    }

private:
    Extensions::Cpu::PagedAttentionFuseConfig m_config;
};

}  // namespace intel_cpu
}  // namespace ov