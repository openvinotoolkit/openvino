// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/storeconvert.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

snippets::op::StoreConvert::StoreConvert(const Output<Node>& x, const ov::element::Type& destination_type, const size_t lanes) :
    Store(x, lanes), m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

bool snippets::op::StoreConvert::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(StoreConvert);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

std::shared_ptr<Node> snippets::op::StoreConvert::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(StoreConvert);
    check_new_args_count(this, new_args);
    return std::make_shared<StoreConvert>(new_args.at(0), m_destination_type, m_lanes);
}

void snippets::op::StoreConvert::validate_and_infer_types() {
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}
