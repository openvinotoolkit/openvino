// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::op {

MOE3GemmFusedCompressed::MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config) : MOECompressed(args, config) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MOE3GemmFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOE3GemmFusedCompressed>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op
