// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/topk_ie.hpp"

#include <memory>
#include <string>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TopKIE::type_info;

op::TopKIE::TopKIE(const Output<Node>& data, const Output<Node>& k, const int64_t axis, const std::string& mode, const std::string& sort,
        const Shape& output_shape)
    : Op({data, k}), axis(axis), mode(mode), sort_type(sort), output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::TopKIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 2) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<TopKIE>(new_args.at(0), new_args.at(1), axis, mode, sort_type, output_shape);
}

void op::TopKIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, element::i32, output_shape);
}

int64_t op::TopKIE::get_axis() {
    return axis;
}

std::string op::TopKIE::get_mode() {
    return mode;
}

std::string op::TopKIE::get_sort_type() {
    return sort_type;
}

Shape op::TopKIE::get_output_shape() {
    return output_shape;
}
