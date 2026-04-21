// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/op/concat.hpp"

namespace ov::intel_cpu::op {

PaKVReorder::PaKVReorder(const Output<Node>& key_cache,
                         const Output<Node>& value_cache,
                         const Output<Node>& block_indices,
                         const Output<Node>& block_indices_begins,
                         const Output<Node>& block_update_indices,
                         const Output<Node>& block_update_indices_begins)
    : Op({key_cache,
          value_cache,
          block_indices,
          block_indices_begins,
          block_update_indices,
          block_update_indices_begins}) {
    constructor_validate_and_infer_types();
}

bool PaKVReorder::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

void PaKVReorder::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 6, "PaKVReorder expects 6 inputs");

    const auto& key_cache_shape = get_input_partial_shape(0);
    const auto& value_cache_shape = get_input_partial_shape(1);

    // Output shape is the concatenation of key and value cache shapes
    // Output is a dummy u8 scalar (placeholder for in-place operation)
    set_output_type(0, ov::element::u8, ov::PartialShape{});
}

std::shared_ptr<Node> PaKVReorder::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PaKVReorder>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         new_args.at(5));
}

}  // namespace ov::intel_cpu::op
