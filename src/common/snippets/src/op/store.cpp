// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/scalarstore.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

snippets::op::Store::Store(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

bool snippets::op::Store::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> snippets::op::Store::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Store);
    check_new_args_count(this, new_args);
    return std::make_shared<Store>(new_args.at(0));
}

void snippets::op::Store::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool snippets::op::Store::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    INTERNAL_OP_SCOPE(Store);
    NGRAPH_CHECK(input_values.size() == this->inputs().size(), "wrong input config");
    NGRAPH_CHECK(output_values.size() == this->outputs().size(), "wrong output config");
    NGRAPH_CHECK(input_values.size() == output_values.size() && input_values.size() == 1, "must be 1->1 operation");
    NGRAPH_CHECK(this->output(0).get_shape() == output_values[0]->get_shape(), "output vector must have the same shape as output port");
    NGRAPH_CHECK(this->input(0).get_shape() == input_values[0]->get_shape(), "input and output must have same shape");
    NGRAPH_CHECK(this->input(0).get_shape() == input_values[0]->get_shape(), "input and output must have same shape");

    std::copy(input_values[0]->get_data_ptr<uint8_t>(),
        input_values[0]->get_data_ptr<uint8_t>() + shape_size(get_output_shape(0))*output_values[0]->get_element_type().size(),
        output_values[0]->get_data_ptr<uint8_t>());

    return true;
}
