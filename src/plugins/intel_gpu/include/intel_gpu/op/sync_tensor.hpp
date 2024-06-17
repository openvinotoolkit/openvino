// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
enum class TP_MODE {
    ALL_REDUCE = 0,
    ALL_GATHERV
};
class SyncTensor : public ov::op::Op {
public:
    OPENVINO_OP("SYNCTENSOR", "gpu_opset");
    SyncTensor() = default;
    SyncTensor(const ov::element::Type output_type);

    SyncTensor(const Output<Node>& input,
           const ov::element::Type output_type = ov::element::undefined,
           const TP_MODE tp_mode = TP_MODE::ALL_REDUCE);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    TP_MODE get_tp_mode() const { return m_tp_mode; }

protected:
    ov::element::Type m_output_type;
    TP_MODE m_tp_mode = TP_MODE::ALL_REDUCE;
};

std::vector<ov::PartialShape> shape_infer(const SyncTensor* op, std::vector<ov::PartialShape> input_shapes);
}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
