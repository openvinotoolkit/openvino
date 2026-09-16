// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/atan2.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/eltwise.hpp"

namespace ov::op::internal {
using Atan2 = ov::intel_gpu::op::Atan2;
}

namespace ov::intel_gpu {

static void CreateAtan2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::Atan2>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto prim = cldnn::eltwise(primitive_name, inputs, cldnn::eltwise_mode::atan2);
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, Atan2);

}  // namespace ov::intel_gpu
