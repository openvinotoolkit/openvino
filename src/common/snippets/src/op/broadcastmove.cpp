// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/broadcastmove.hpp"


namespace ov {
namespace snippets {
namespace op {

BroadcastMove::BroadcastMove(const Output<Node>& x, ov::Dimension bcast_dimension) : Op({x}), bcast_dimension(std::move(bcast_dimension)) {
    constructor_validate_and_infer_types();
}

bool BroadcastMove::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("bcast_dimension", bcast_dimension);
    return true;
}

std::shared_ptr<Node> BroadcastMove::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BroadcastMove);
    check_new_args_count(this, new_args);
    return std::make_shared<BroadcastMove>(new_args.at(0), bcast_dimension);
}

void BroadcastMove::validate_and_infer_types() {
    auto broadcasted_shape = get_input_partial_shape(0);
    if (broadcasted_shape.size() == 0)
        broadcasted_shape.resize(1);
    *broadcasted_shape.rbegin() = bcast_dimension;
    set_output_type(0, get_input_element_type(0), broadcasted_shape);
}

} // namespace op
} // namespace snippets
} // namespace ov
