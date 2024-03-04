// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class GatherCompressionNode : public ov::op::Op {
public:
    OPENVINO_OP("GatherCompression", "cpu_plugin_opset");

    GatherCompressionNode() = default;

    GatherCompressionNode(const ov::Output<Node>& data,
                          const ov::Output<Node>& zp_compressed,
                          const ov::Output<Node>& scale_compressed,
                          const ov::Output<Node>& indices,
                          const ov::Output<Node>& axis,
                          const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

private:
    ov::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
