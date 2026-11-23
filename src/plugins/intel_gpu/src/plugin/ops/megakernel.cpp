// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/megakernel.hpp"

// Alias into the ov::op::internal namespace so REGISTER_FACTORY_IMPL can find it.
namespace ov {
namespace op {
namespace internal {
using MegaKernel = ov::intel_gpu::op::MegaKernel;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMegaKernelOp(ProgramBuilder& p,
                                      const std::shared_ptr<ov::intel_gpu::op::MegaKernel>& op) {
    std::vector<cldnn::input_info> prim_inputs;
    for (auto& ii : p.GetInputInfo(op))
        prim_inputs.push_back(cldnn::input_info(ii));

    auto prim = cldnn::megakernel(layer_type_name_ID(op), prim_inputs, op->get_attrs());
    prim.output_data_types = get_output_data_types(op);
    // The megakernel writes its declared types, so the plugin must not substitute another one.
    for (size_t i = 0; i < prim.output_data_types.size(); ++i) {
        OPENVINO_ASSERT(prim.output_data_types[i].value_or(ov::element::dynamic) == op->get_output_element_type(i),
                        "[MegaKernel] output ", i, " type ", op->get_output_element_type(i), " is not supported by the GPU plugin");
    }
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, MegaKernel);

}  // namespace ov::intel_gpu
