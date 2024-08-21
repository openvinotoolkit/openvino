// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/constant.hpp"

#include "util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
class RankConstant : public ov::op::v0::Constant {
public:
    OPENVINO_OP("RankConstant", "gpu_opset");
    RankConstant(const std::shared_ptr<ov::Node>& other, const size_t world_size, const size_t world_rank);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    TP_MODE get_tp_mode() const { return m_tp_mode; }
    int get_size() const { return m_world_size; }
    int get_rank() const { return m_world_rank; }
protected:
    ov::element::Type m_output_type;
    int m_world_size;
    int m_world_rank;
    Shape m_shape{};
    element::Type m_element_type{};
    TP_MODE m_tp_mode = TP_MODE::ALL_GATHERH;
};

std::vector<ov::PartialShape> shape_infer(const RankConstant* op, std::vector<ov::PartialShape> input_shapes);
}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
