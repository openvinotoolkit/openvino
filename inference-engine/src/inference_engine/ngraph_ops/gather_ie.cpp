// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_ie.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GatherIE::type_info;

op::GatherIE::GatherIE(const Output<Node>& params, const Output<Node>& indices, int64_t axis, const Shape & output_shape)
        : Op({params, indices})
        , m_axis(axis)
        , m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GatherIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<GatherIE>(new_args.at(0), new_args.at(1), m_axis, m_output_shape);
}

void op::GatherIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_output_shape);
}
