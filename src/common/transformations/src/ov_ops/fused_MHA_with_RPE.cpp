// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fused_MHA_with_RPE.hpp"

#include "itt.hpp"

using namespace std;
using namespace ov::op::internal;

FusedMHA_RPE::FusedMHA_RPE(const Output<Node>& data,
                           const Output<Node>& sin,
                           const Output<Node>& cos,
                           const Output<Node>& prev_keys,
                           const Output<Node>& prev_values,
                           const Output<Node>& boolean_mask,
                           const Output<Node>& attention_mask,
                           const Output<Node>& keys_multiplier,
                           const size_t& dhead)
    : Op({data, sin, cos, prev_keys, prev_values, boolean_mask, attention_mask, keys_multiplier}),
      d_head{dhead} {
    constructor_validate_and_infer_types();
}

void FusedMHA_RPE::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_FusedMHA_RPE_validate_and_infer_types);
    const auto& input_shape = get_input_partial_shape(0);
    OPENVINO_ASSERT(input_shape.rank().is_static() && input_shape.size() == 4);
    auto output_shape = PartialShape::dynamic(3);
    output_shape[0] = input_shape[0];
    output_shape[1] = input_shape[2];
    output_shape[2] = input_shape[1] * d_head;
    set_output_type(0, get_input_element_type(0), output_shape);
}

bool FusedMHA_RPE::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_FusedMHA_RPE_visit_attributes);
    visitor.on_attribute("d_head", d_head);
    return true;
}

shared_ptr<ov::Node> FusedMHA_RPE::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_FusedMHA_RPE_clone_with_new_inputs);
    return make_shared<FusedMHA_RPE>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     new_args.at(5),
                                     new_args.at(6),
                                     new_args.at(7),
                                     d_head);
}
