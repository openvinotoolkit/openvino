// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/gather_ie.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GatherIE::type_info;

op::GatherIE::GatherIE(const Output<Node>& params, const Output<Node>& indices, int64_t axis)
        : Op({params, indices})
        , m_axis(axis) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GatherIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<GatherIE>(new_args.at(0), new_args.at(1), m_axis);
}

void op::GatherIE::validate_and_infer_types() {
    // Use opset1::Gather to calculate output shape
    auto gather = std::make_shared<opset1::Gather>(input_value(0), input_value(1), opset1::Constant::create(element::i64, Shape{1}, {m_axis}));
    set_output_type(0, gather->output(0).get_element_type(), gather->output(0).get_partial_shape());
}
