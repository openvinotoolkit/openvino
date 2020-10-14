// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/eltwise.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Eltwise::type_info;

op::Eltwise::Eltwise(const Output<Node>& data1, const Output<Node>& data2, const ELTWISE_TYPE eltwise_type, const element::Type output_type)
    : Op({data1, data2}), eltwise_type(eltwise_type), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::Eltwise::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 2) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Eltwise>(new_args.at(0), new_args.at(1), eltwise_type, m_output_type);
}

void op::Eltwise::validate_and_infer_types() {
    //  Check that weights and biases has the same type
    element::Type data1_et = get_input_element_type(0);
    element::Type data2_et = get_input_element_type(1);

    element::Type et_result;
    if (m_output_type == element::undefined) {
        NODE_VALIDATION_CHECK(this, element::Type::merge(et_result, data1_et, data2_et),
                              "Element types for first and second do not match :", data1_et, " and ", data2_et);
    } else {
        et_result = m_output_type;
    }

    if (get_input_partial_shape(0).rank().is_dynamic() ||
        get_input_partial_shape(1).rank().is_dynamic()) {
        set_output_type(0, et_result, PartialShape::dynamic());
        return;
    }

    std::vector<Dimension> shape1(get_input_partial_shape(0));
    std::vector<Dimension> shape2(get_input_partial_shape(1));

    std::vector<Dimension> output_shape(PartialShape::dynamic(std::max(shape1.size(), shape2.size())));
    auto output_shape_it = output_shape.rbegin();

    auto shape1_it = shape1.rbegin(), shape2_it = shape2.rbegin();
    while (shape1_it != shape1.rend() || shape2_it != shape2.rend()) {
        if (shape1_it != shape1.rend() && shape2_it != shape2.rend()) {
            if (shape1_it->is_static() && shape2_it->is_static()) {
                *output_shape_it = (shape1_it->get_length() > shape2_it->get_length() ? *shape1_it : *shape2_it);
            }
        } else if (shape1_it != shape1.rend()) {
            *output_shape_it = *shape1_it;
        } else if (shape2_it != shape2.rend()) {
            *output_shape_it = *shape2_it;
        }

        if (shape1_it != shape1.rend()) ++shape1_it;
        if (shape2_it != shape2.rend()) ++shape2_it;
        ++output_shape_it;
        if (output_shape_it == output_shape.rend()) {
            break;
        }
    }

    set_output_type(0, et_result, output_shape);
}
