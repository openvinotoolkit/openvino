// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm.hpp"

#include <memory>
#include <numeric>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GemmIE::type_info;

op::GemmIE::GemmIE(const Output<Node>& A, const Output<Node>& B, const bool& transpose_a, const bool& transpose_b,
                   const Shape& output_shape)
    : Op({A, B}), m_transpose_a(transpose_a), m_transpose_b(transpose_b), m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GemmIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<GemmIE>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b, m_output_shape);
}

void op::GemmIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_output_shape);
}
