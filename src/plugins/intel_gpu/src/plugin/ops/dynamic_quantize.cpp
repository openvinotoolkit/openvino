// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"

namespace ov {
namespace op {
namespace internal {
using DynamicQuantize = ov::intel_gpu::op::DynamicQuantize;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<op::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    OPENVINO_ASSERT(op->get_group_size() == 1048576, "Not supported group size: ", op->get_group_size());
    auto prim = cldnn::dynamic_quantize(primitive_name,
                                inputs[0],
                                op->get_group_size(),
                                get_output_data_types(op)
                                );
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);

}  // namespace intel_gpu
}  // namespace ov
