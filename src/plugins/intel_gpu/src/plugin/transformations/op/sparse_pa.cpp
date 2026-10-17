// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

namespace ov::intel_gpu::op {

std::shared_ptr<Node> SparsePA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SparsePA>(new_args, get_config(), get_output_type());
}

}  // namespace ov::intel_gpu::op
