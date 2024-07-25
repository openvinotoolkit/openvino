// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rank_constant.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

RankConstant::RankConstant(const std::shared_ptr<ov::Node>& constant_data,
            const size_t world_size,
            const size_t world_rank) : ov::op::v0::Constant(*std::dynamic_pointer_cast<ov::op::v0::Constant>(constant_data)),
            m_world_size(world_size),
            m_world_rank(world_rank) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(constant_data);
    m_shape = constant->get_shape();
    m_element_type = constant->get_element_type();
    // adjusting the shape here for graph validating
    int split_dim = 0;
    auto split_parts = [](int len, int n) {
        int average = len / n;
        std::vector<int> parts(n, average);
        parts.back() = len - average * (n - 1);
        return parts;
    };
    auto split_dims = split_parts(m_shape[split_dim], m_world_size);
    m_shape[split_dim] = split_dims.at(m_world_rank);
    validate_and_infer_types();
}

bool RankConstant::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("element_type", m_element_type);
    visitor.on_attribute("shape", m_shape);
    return true;
}

void RankConstant::validate_and_infer_types() {
    set_output_type(0, m_element_type, m_shape);
}

std::shared_ptr<Node> RankConstant::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Constant>(*this);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
