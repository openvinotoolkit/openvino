// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/gather.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class GatherCompressed : public ov::op::v8::Gather {
public:
    OPENVINO_OP("GatherCompressed", "gpu_opset");

    GatherCompressed() = default;

    GatherCompressed(const ov::Output<Node> &data,
                     const ov::Output<Node> &indices,
                     const ov::Output<Node> &axis,
                     const ov::Output<Node> &decompression_scale,
                     const ov::Output<Node> &decompression_zero_point,
                     const ov::element::Type output_type = ov::element::undefined);

    GatherCompressed(const ov::Output<Node> &data,
                     const ov::Output<Node> &indices,
                     const ov::Output<Node> &axis,
                     const ov::Output<Node> &decompression_scale,
                     const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    ov::element::Type m_output_type;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
