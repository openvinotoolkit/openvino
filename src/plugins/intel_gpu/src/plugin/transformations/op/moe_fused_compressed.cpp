// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_fused_compressed.hpp"

namespace ov::intel_gpu::op {

MOEFusedCompressed::MOEFusedCompressed(const OutputVector& args, const MOECompressed::Config config) : MOECompressed(args, config) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MOEFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOEFusedCompressed>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op
