// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa.hpp"

namespace ov::intel_gpu::op {

// Config validation, visit_attributes() and validate_and_infer_types() are inherited
// unchanged from ov::op::internal::QSA (see src/core/src/op/qsa.cpp); only
// clone_with_new_inputs() needs to be overridden so that clones stay this (gpu_opset)
// derived type instead of the base internal::QSA type.
std::shared_ptr<Node> QSA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSA>(new_args, get_config(), get_output_type());
}

}  // namespace ov::intel_gpu::op

