// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/truncation.hpp"

using namespace std;
using namespace ngraph;

snippets::op::Truncation::Truncation(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

void snippets::op::Truncation::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Truncation);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Only accepts one argument. Got: ", get_input_size());
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> snippets::op::Truncation::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Truncation);
    check_new_args_count(this, new_args);
    auto other = std::make_shared<Truncation>(new_args.at(0));
    return other;
}
