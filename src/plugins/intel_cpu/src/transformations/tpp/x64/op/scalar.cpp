// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scalar.hpp"
#include "modifiers.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

Scalar::Scalar(const snippets::op::Scalar& other) : ov::snippets::op::Scalar(other) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Scalar::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<tpp::op::Scalar>(*this);
}

bool Scalar::visit_attributes(AttributeVisitor& visitor) {
    modifier::TensorProcessingPrimitive::visit_attributes(visitor);
    return  snippets::op::Scalar::visit_attributes(visitor);;
}


} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
