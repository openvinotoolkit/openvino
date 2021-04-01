// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/topk_ie.hpp"

#include <memory>
#include <string>
#include <ngraph/opsets/opset1.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TopKIE::type_info;


op::TopKIE::TopKIE(const ngraph::Output<ngraph::Node> &data, const ngraph::Output<ngraph::Node> &k, const int64_t axis, const ngraph::op::TopKMode mode,
                   const ngraph::op::TopKSortType sort, const element::Type& index_element_type)
    : Op({data, k}), m_axis(axis), m_mode(mode), m_sort_type(sort), m_index_element_type(index_element_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::TopKIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<TopKIE>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort_type, m_index_element_type);
}

void op::TopKIE::validate_and_infer_types() {
    const auto& input_partial_shape = get_input_partial_shape(0);
    const auto input_rank = input_partial_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() > 0,
                          "Input rank must be greater than 0.");

    const auto& k_partial_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, k_partial_shape.rank().compatible(1), "The 'K' input must be a 1D tensor.");

    // Construct v1::TopK operation to calculate output shapes
    std::shared_ptr<Node> topk;
    if (auto k_const = std::dynamic_pointer_cast<opset1::Constant>(input_value(1).get_node_shared_ptr())) {
        const auto k = k_const->cast_vector<int64_t>();
        topk = std::make_shared<opset1::TopK>(input_value(0),
                                              opset1::Constant::create(element::i64, Shape{}, k),
                                              m_axis, m_mode, m_sort_type, m_index_element_type);
    } else {
        topk = std::make_shared<opset1::TopK>(input_value(0),
                                              std::make_shared<opset1::Squeeze>(input_value(1), opset1::Constant::create(element::i64, Shape{1}, {0})),
                                              m_axis, m_mode, m_sort_type, m_index_element_type);
    }

    set_output_size(2);
    set_output_type(0, topk->get_output_element_type(0), topk->get_output_partial_shape(0));
    set_output_type(1, topk->get_output_element_type(1), topk->get_output_partial_shape(1));
}

bool op::TopKIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("sort", m_sort_type);
    return true;
}
