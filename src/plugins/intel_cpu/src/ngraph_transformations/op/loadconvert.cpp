// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "loadconvert.hpp"

#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ov;

intel_cpu::LoadConvert::LoadConvert(const Output<Node>& x, const ov::element::Type& destination_type, const size_t count) :
    Load(x, count), m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

bool intel_cpu::LoadConvert::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(LoadConvert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

std::shared_ptr<Node> intel_cpu::LoadConvert::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LoadConvert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LoadConvert>(new_args.at(0), m_destination_type, m_count);
}

void intel_cpu::LoadConvert::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(LoadConvert_validate_and_infer_types);
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}
