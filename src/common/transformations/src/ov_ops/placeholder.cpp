// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/placeholder.hpp"

namespace ov {
namespace op {
namespace internal {

Placeholder::Placeholder() : ov::op::Op() {
    validate_and_infer_types();
}

void Placeholder::validate_and_infer_types() {
    set_output_type(0, ov::element::undefined, ov::PartialShape{});
}

std::shared_ptr<Node> Placeholder::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Placeholder>();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
