// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/broadcastload.hpp"

#include <ngraph/runtime/reference/broadcast.hpp>

using namespace std;
using namespace ngraph;

snippets::op::BroadcastLoad::BroadcastLoad(const Output<Node>& x, ov::PartialShape shape, size_t offset)
    : BroadcastMove(x, std::move(shape)), m_offset(offset) {
    constructor_validate_and_infer_types();
}

bool snippets::op::BroadcastLoad::visit_attributes(AttributeVisitor& visitor) {
    BroadcastMove::visit_attributes(visitor);
    visitor.on_attribute("offset", m_offset);
    return true;
}

std::shared_ptr<Node> snippets::op::BroadcastLoad::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BroadcastLoad);
    check_new_args_count(this, new_args);
    return std::make_shared<BroadcastLoad>(new_args.at(0), output_shape, m_offset);
}

void snippets::op::BroadcastLoad::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), output_shape);
}
