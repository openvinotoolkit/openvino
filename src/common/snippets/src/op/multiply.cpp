// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/multiply.hpp"
#include <ngraph/runtime/host_tensor.hpp>
#include <snippets/itt.hpp>

using namespace std;
using namespace ngraph;

snippets::op::Multiply::Multiply(
    const Output<Node>& arg0,
    const Output<Node>& arg1,
    const ngraph::op::AutoBroadcastSpec& auto_broadcast) : ngraph::opset1::Multiply(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

void snippets::op::Multiply::validate_and_infer_types() {
    const auto input_type1 = get_input_element_type(0);
    const auto input_type2 = get_input_element_type(1);
    
    // TODO: not completed
    const element::Type output_type = (input_type1 == element::i8) || (input_type2 == element::i8) ? 
        element::i32 : 
        get_input_element_type(0);

    set_output_type(0, output_type, get_input_partial_shape(0));
}

bool snippets::op::Multiply::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    // TODO: not completed
    return true;
}
