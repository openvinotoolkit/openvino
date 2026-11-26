// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_post.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/hyper_connection_post.hpp"

namespace ov::intel_gpu {

static void CreateHyperConnectionPostOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::HyperConnectionPost>& op) {
    validate_inputs_count(op, {4});
    cldnn::hyper_connection_post prim(layer_type_name_ID(op), p.GetInputInfo(op));
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, HyperConnectionPost);

}  // namespace ov::intel_gpu
