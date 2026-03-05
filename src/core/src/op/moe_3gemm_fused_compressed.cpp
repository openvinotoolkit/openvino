// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/moe_3gemm_fused_compressed.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MOE3GemmFusedCompressed::MOE3GemmFusedCompressed(const OutputVector& args, const Config config)
    : MOECompressed(args, config) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MOE3GemmFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_MOE3GemmFusedCompressed_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<MOE3GemmFusedCompressed>(new_args, get_config());
}

}  // namespace internal
}  // namespace op
}  // namespace ov
