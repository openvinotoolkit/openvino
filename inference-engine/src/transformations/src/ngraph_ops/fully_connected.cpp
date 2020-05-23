// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/fully_connected.hpp"

#include <memory>
#include <numeric>

#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::FullyConnected::type_info;

op::FullyConnected::FullyConnected(const Output<Node>& A, const Output<Node>& B, const Output<Node>& C, const Shape & output_shape)
    : Op({A, B, C}), m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::FullyConnected::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<FullyConnected>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_shape);
}

void op::FullyConnected::validate_and_infer_types() {
    if (m_output_shape.size() < 2)
        throw ngraph_error("FullyConnected shape is incorrect");
    m_output_size = m_output_shape.back();
    set_output_type(0, input_value(0).get_element_type(), m_output_shape);
}
