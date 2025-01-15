// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_mul_add.hpp"

#include "openvino/op/util/elementwise_args.hpp"
#include "snippets/itt.hpp"

using namespace ov;
using namespace ov::intel_cpu;

FusedMulAdd::FusedMulAdd(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c) : Op({a, b, c}) {
    constructor_validate_and_infer_types();
}

bool FusedMulAdd::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> FusedMulAdd::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FusedMulAdd);
    check_new_args_count(this, new_args);
    return std::make_shared<FusedMulAdd>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void FusedMulAdd::validate_and_infer_types() {
    const auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 3, "FusedMulAdd must have 3 inputs");
    OPENVINO_ASSERT(get_output_size() == 1, "FusedMulAdd must have only 1 output");

    const auto element_type = get_input_element_type(0);
    auto pshape = get_input_partial_shape(0);
    for (size_t i = 1; i < input_size; ++i) {
        NODE_VALIDATION_CHECK(this,
                              element_type == get_input_element_type(i),
                              "Argument element types are inconsistent.");
        NODE_VALIDATION_CHECK(
            this,
            PartialShape::broadcast_merge_into(pshape, get_input_partial_shape(i), ov::op::AutoBroadcastType::NUMPY),
            "Argument shapes are inconsistent.");
    }
    set_output_type(0, element_type, pshape);
}

const ov::op::AutoBroadcastSpec& FusedMulAdd::get_autob() const {
    static ov::op::AutoBroadcastSpec autob_spec(ov::op::AutoBroadcastType::NUMPY);
    return autob_spec;
}
