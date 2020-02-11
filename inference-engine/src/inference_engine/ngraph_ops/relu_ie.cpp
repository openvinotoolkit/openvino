// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "relu_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReLUIE::type_info;

op::ReLUIE::ReLUIE(const Output<Node>& data, const float& negative_slope)
    : Op(OutputVector {data}), m_negative_slope(negative_slope) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::ReLUIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ReLUIE>(new_args.at(0), m_negative_slope);
}

void op::ReLUIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
