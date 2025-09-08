// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/conditional_block.hpp"

#include <algorithm>

#include "openvino/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

op::internal::ConditionalBlock::ConditionalBlock(
    const Output<Node>& cond,
    bool is_scalar_condition,
    int32_t sub_block_index,
    const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
    : Op({cond}),
      m_is_scalar_condition(is_scalar_condition),
      m_sub_block_index(sub_block_index),
      m_output_infos(output_infos) {
    constructor_validate_and_infer_types();
}

op::internal::ConditionalBlock::ConditionalBlock(
    const OutputVector& inputs,
    const Output<Node>& cond,
    bool is_scalar_condition,
    int32_t sub_block_index,
    const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
    : m_is_scalar_condition(is_scalar_condition),
      m_sub_block_index(sub_block_index),
      m_output_infos(output_infos) {
    OutputVector new_args;
    std::move(inputs.begin(), inputs.end(), std::back_inserter(new_args));
    new_args.emplace_back(cond);
    set_arguments(new_args);
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::ConditionalBlock::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 1) {  // w/o inputs
        return make_shared<ConditionalBlock>(new_args.at(0), m_is_scalar_condition, m_sub_block_index, m_output_infos);
    } else {
        OutputVector inputs_args;
        for (size_t i = 0; i < new_args.size() - 1; i++) {
            inputs_args.push_back(new_args[i]);
        }
        return make_shared<ConditionalBlock>(inputs_args,
                                             new_args.back(),
                                             m_is_scalar_condition,
                                             m_sub_block_index,
                                             m_output_infos);
    }
}

bool op::internal::ConditionalBlock::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("is_scalar_condition", m_is_scalar_condition);
    visitor.on_attribute("sub_block_index", m_sub_block_index);
    return true;
}

void op::internal::ConditionalBlock::validate_and_infer_types() {
    for (size_t i = 0; i < m_output_infos.size(); i++) {
        set_output_type(i, m_output_infos[i].first, m_output_infos[i].second);
    }
}

const OutputVector op::internal::ConditionalBlock::get_inputs_from_parent() const {
    OutputVector result;
    const auto& inputs = this->input_values();
    for (size_t i = 0; i < inputs.size() - 1; i++) {  // except the one at last, which is "cond".
        result.push_back(inputs[i]);
    }
    return result;
}
