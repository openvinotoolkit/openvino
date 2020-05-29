// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/power.hpp"

#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PowerIE::type_info;

op::PowerIE::PowerIE(const Output<ngraph::Node>& data_batch, const float power, const float scale, const float shift)
    : Op({data_batch}), scale(scale), power(power), shift(shift) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::PowerIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<PowerIE>(new_args.at(0), this->power, this->scale, this->shift);
}

void op::PowerIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
