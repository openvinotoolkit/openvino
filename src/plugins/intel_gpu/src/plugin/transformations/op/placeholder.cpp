// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov::intel_gpu::op {

Placeholder::Placeholder() : ov::op::Op() {
    validate_and_infer_types();
    set_friendly_name(get_name());
    get_rt_info().emplace(FusedNames::get_type_info_static(), FusedNames{get_friendly_name()});
}

bool Placeholder::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

void Placeholder::validate_and_infer_types() {
    set_output_type(0, ov::element::dynamic, ov::PartialShape{});
}

std::shared_ptr<Node> Placeholder::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Placeholder>();
}

}  // namespace ov::intel_gpu::op
