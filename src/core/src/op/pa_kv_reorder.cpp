// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pa_kv_reorder.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "pa_kv_reorder_shape_inference.hpp"

namespace ov::op::internal {

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

bool PaKVReorder::visit_attributes(ov::AttributeVisitor& /*visitor*/) {
    return true;
}

void PaKVReorder::validate_and_infer_types() {
    OV_OP_SCOPE(PaKVReorder_validate_and_infer_types);

    for (size_t i = 2; i < 6; ++i) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(i).is_dynamic() || get_input_element_type(i) == ov::element::i32,
                              "Input ",
                              i,
                              " (indices) must be i32, got ",
                              get_input_element_type(i));
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, ov::element::u8, output_shapes[0]);
}

std::shared_ptr<Node> PaKVReorder::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PaKVReorder_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PaKVReorder>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         new_args.at(5));
}

}  // namespace ov::op::internal
