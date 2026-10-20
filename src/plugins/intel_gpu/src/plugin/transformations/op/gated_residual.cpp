// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gated_residual.hpp"

namespace ov::intel_gpu::op {

// visit_attributes() and validate_and_infer_types() are inherited unchanged from
// ov::op::internal::GatedResidual (see src/core/src/op/gated_residual.cpp); only
// clone_with_new_inputs() needs to be overridden so that clones stay this (gpu_opset)
// derived type instead of the base internal::GatedResidual type.
std::shared_ptr<Node> GatedResidual::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GatedResidual>(new_args,
                                           get_mode(),
                                           get_hc_count(),
                                           get_hidden_size(),
                                           get_lowrank(),
                                           get_epsilon(),
                                           get_norm_gain_plus_one(),
                                           get_produce_write_gate(),
                                           get_output_type());
}

}  // namespace ov::intel_gpu::op

