// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include <memory>
#include <numeric>

#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::FullyConnected::type_info;

op::FullyConnected::FullyConnected(const Output<Node>& A, const Output<Node>& B, const Output<Node>& C): Op({A, B, C}) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::FullyConnected::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<FullyConnected>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::FullyConnected::validate_and_infer_types() {
    element::Type result_et;
    NODE_VALIDATION_CHECK(this, element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ", get_input_element_type(0),
                          ", arg1 element type: ", get_input_element_type(1), ").");

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);

    if (arg0_shape.is_dynamic() || arg1_shape.is_dynamic()) {
        set_output_type(0, result_et, PartialShape::dynamic());
    } else {
        // FullyConnected representation: [I, K] * [O, K] = [I, O]
        Shape a = arg0_shape.to_shape();
        Shape b = arg1_shape.to_shape();
        set_output_type(0, result_et, Shape {a[0], b[0]});
        m_output_size = b[0];
    }
}