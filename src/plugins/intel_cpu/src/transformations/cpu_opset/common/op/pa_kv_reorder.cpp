// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

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
    OPENVINO_ASSERT(get_input_size() == 6, "PaKVReorder expects 6 inputs");

    const auto key_type = get_input_element_type(0);
    const auto value_type = get_input_element_type(1);
    const auto is_supported_cache_type = [](const ov::element::Type& type) {
        return type.is_dynamic() || type == ov::element::f32 || type == ov::element::bf16 || type == ov::element::f16 ||
               type == ov::element::u8 || type == ov::element::u4;
    };

    NODE_VALIDATION_CHECK(this,
                          is_supported_cache_type(key_type),
                          "PaKVReorder supports only f32, bf16, f16, u8, and u4 KV cache precisions. ",
                          "int8 key_cache is not supported.");
    NODE_VALIDATION_CHECK(this,
                          is_supported_cache_type(value_type),
                          "PaKVReorder supports only f32, bf16, f16, u8, and u4 KV cache precisions. ",
                          "int8 value_cache is not supported.");

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

}  // namespace ov::intel_cpu
