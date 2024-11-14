// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/reduce.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"

namespace ov {
namespace snippets {
namespace op {
ReduceBase::ReduceBase(const Output<Node>& x, size_t axis) : Op({x}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool ReduceBase::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    return true;
}

void ReduceBase::validate_and_infer_types() {
    auto result_shape = get_input_partial_shape(0);
    result_shape[m_axis] = 1;
    set_output_type(0, get_input_element_type(0), result_shape);
}

void ReduceBase::compute_and_set_reduce_subtensors(const std::shared_ptr<ReduceBase>& reduce) {
    OPENVINO_ASSERT(reduce->get_input_partial_shape(0).rank().is_static(),
                    "Subtensors can be automatically calculated only for reduce with static rank.");
    const auto reduce_rank = reduce->get_input_partial_shape(0).size();
    const auto axis = reduce->get_axis();

    std::vector<size_t> subtensor(reduce_rank, 1);
    for (size_t i = axis; i < reduce_rank; ++i)
        subtensor[i] = utils::get_full_dim_value();
    lowered::PortDescriptorUtils::set_port_descriptor_ptr(reduce->input(0), std::make_shared<lowered::PortDescriptor>(reduce->input(0), subtensor));
    lowered::PortDescriptorUtils::set_port_descriptor_ptr(reduce->output(0), std::make_shared<lowered::PortDescriptor>(reduce->output(0), subtensor));
}

std::shared_ptr<Node> ReduceSum::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ReduceSum);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceSum>(new_args.at(0), m_axis);
}

std::shared_ptr<Node> ReduceMax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ReduceMax);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceMax>(new_args.at(0), m_axis);
}

} // namespace op
} // namespace snippets
} // namespace ov
