// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "openvino/op/placeholder_extension.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {

PlaceholderExtension::PlaceholderExtension() : ov::op::Op() {
    validate_and_infer_types();
    set_friendly_name(get_name());
    get_rt_info().emplace(FusedNames::get_type_info_static(), FusedNames{get_friendly_name()});
}

bool PlaceholderExtension::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_PlaceholderExtension_visit_attributes);
    return true;
}

void PlaceholderExtension::validate_and_infer_types() {
    OV_OP_SCOPE(internal_PlaceholderExtension_validate_and_infer_types);
    set_output_type(0, ov::element::dynamic, ov::PartialShape{});
}

std::shared_ptr<Node> PlaceholderExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_PlaceholderExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PlaceholderExtension>();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
