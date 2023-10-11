// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/broadcastmove.hpp"


namespace ov {
namespace snippets {
namespace op {

BroadcastMove::BroadcastMove(const Output<Node>& x, ov::PartialShape shape) : Op({x}), output_shape(std::move(shape)) {
    constructor_validate_and_infer_types();
}

bool BroadcastMove::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("output_shape", output_shape);
    return true;
}

std::shared_ptr<Node> BroadcastMove::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BroadcastMove);
    check_new_args_count(this, new_args);
    return std::make_shared<BroadcastMove>(new_args.at(0), output_shape);
}

void BroadcastMove::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), this->output_shape);
}

} // namespace op
} // namespace snippets
} // namespace ov
