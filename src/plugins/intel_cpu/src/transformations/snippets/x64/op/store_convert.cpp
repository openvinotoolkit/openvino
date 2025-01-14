// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "store_convert.hpp"

#include "snippets/itt.hpp"

using namespace std;
using namespace ov;

intel_cpu::StoreConvertSaturation::StoreConvertSaturation(const Output<Node>& x,
                                                          const ov::element::Type& destination_type,
                                                          const size_t count,
                                                          const size_t offset)
    : Store(x, count, offset),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

bool intel_cpu::StoreConvertSaturation::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(StoreConvert_visit_attributes);
    Store::visit_attributes(visitor);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

void intel_cpu::StoreConvertSaturation::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(StoreConvert_validate_and_infer_types);
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> intel_cpu::StoreConvertSaturation::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(StoreConvert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<StoreConvertSaturation>(new_args.at(0), m_destination_type, get_count(), get_offset());
}

intel_cpu::StoreConvertTruncation::StoreConvertTruncation(const Output<Node>& x,
                                                          const ov::element::Type& destination_type,
                                                          const size_t count,
                                                          const size_t offset)
    : Store(x, count, offset),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

bool intel_cpu::StoreConvertTruncation::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(StoreConvert_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

void intel_cpu::StoreConvertTruncation::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(StoreConvert_validate_and_infer_types);
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> intel_cpu::StoreConvertTruncation::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(StoreConvert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<StoreConvertTruncation>(new_args.at(0), m_destination_type, get_count(), get_offset());
}
