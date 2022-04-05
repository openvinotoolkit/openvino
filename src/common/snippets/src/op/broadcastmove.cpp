// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/broadcastmove.hpp"

#include <ngraph/runtime/host_tensor.hpp>
#include <ngraph/runtime/reference/broadcast.hpp>

using namespace std;
using namespace ngraph;

snippets::op::BroadcastMove::BroadcastMove(const Output<Node>& x, Shape shape) : Op({x}), output_shape(shape) {
    constructor_validate_and_infer_types();
}

bool snippets::op::BroadcastMove::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> snippets::op::BroadcastMove::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BroadcastMove);
    check_new_args_count(this, new_args);
    auto other = std::make_shared<BroadcastMove>(new_args.at(0), this->output_shape);
    return other;
}

void snippets::op::BroadcastMove::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), this->output_shape);
}

bool snippets::op::BroadcastMove::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    INTERNAL_OP_SCOPE(BroadcastMove);
    NGRAPH_CHECK(input_values.size() == this->inputs().size(), "wrong input config");
    NGRAPH_CHECK(output_values.size() == this->outputs().size(), "wrong output config");
    NGRAPH_CHECK(input_values.size() == output_values.size() && input_values.size() == 1, "must be 1->1 operation");
    NGRAPH_CHECK(this->output(0).get_shape() == output_values[0]->get_shape(), "output vector must have the same shape as output port");
    NGRAPH_CHECK(this->input(0).get_shape() == input_values[0]->get_shape(), "input and output must have same shape");

    auto ishape = input_values[0]->get_shape();
    auto oshape = output_values[0]->get_shape();

    NGRAPH_CHECK(ishape.size() == oshape.size(), "input and output should have the same rank");

    AxisSet broadcast_axes;
    for (size_t k = 0; k < ishape.size(); k++) {
        if (!((ishape[k] == oshape[k])
           || (ishape[k] != oshape[k] && ((ishape[k] == 1) != (oshape[k] == 1) ) ))) {
            throw ngraph_error("FakeBroadcast::evaluate incompatible shapes");
        }

        if (ishape[k] != oshape[k]) {
            broadcast_axes.insert(k);
        }
    }

    runtime::reference::broadcast(input_values[0]->get_data_ptr<char>(),
                                  output_values[0]->get_data_ptr<char>(),
                                  input_values[0]->get_shape(),
                                  output_values[0]->get_shape(),
                                  broadcast_axes,
                                  sizeof(float));
    return true;
}