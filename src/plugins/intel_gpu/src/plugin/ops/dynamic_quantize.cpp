// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto group_size = op->get_group_size();
    for (size_t i = 0; i < group_size.size() - 1; i++)
        OPENVINO_ASSERT(group_size[i] == 1, "Not supported group size at ", i, ": ", group_size[i]);

    OPENVINO_ASSERT(group_size.back() == UINT64_MAX, "Not supported group size: ", group_size.back());
    auto prim = cldnn::dynamic_quantize(primitive_name,
                                inputs[0],
                                op->get_group_size().back(),
                                get_output_data_types(op)
                                );
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);

}  // namespace intel_gpu
}  // namespace ov
