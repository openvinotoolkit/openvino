// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class RoPENode : public ngraph::op::Op {
public:
    OPENVINO_OP("RoPE", "cpu_plugin_opset");

    RoPENode() = default;

    struct Config {
        size_t slice_start = 0;         // slice inner-most dimensions of input
        size_t slice_stop = 0;
        bool input_trans0213 = false;   // transpose input before *cos/*sin
        bool cos_is_raw3d = false;      // cos input is [B,L,ndims/2]
        bool sin_is_raw3d = false;      // sin input is [B,L,ndims/2]
        bool output_trans0213 = false;  // true implies that input is [B,L,H,S]
        size_t ndims = 0;
        int gather_position_arg_id = 0;
        int concat_with_past_arg_id = 0;
    };

    RoPENode(const OutputVector& args, const Config& cfg);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    Config& get_config() {
        return m_config;
    }

private:
    Config m_config;
};

}   // namespace intel_cpu
}   // namespace ov
