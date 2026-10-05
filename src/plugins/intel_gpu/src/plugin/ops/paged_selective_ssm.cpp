// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include <intel_gpu/primitives/paged_selective_ssm.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

static void CreatePagedSelectiveSSMOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedSelectiveSSM>& op) {
    validate_inputs_count(op, {11});
    auto inputs = p.GetInputInfo(op);

    const std::string layer_name = layer_type_name_ID(op);
    cldnn::paged_selective_ssm prim(layer_name, inputs);
    prim.num_outputs = 1;
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedSelectiveSSM);

}  // namespace ov::intel_gpu
