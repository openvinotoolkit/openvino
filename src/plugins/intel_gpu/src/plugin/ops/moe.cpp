// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_fused_compressed.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_fused_compressed.hpp"
#include "intel_gpu/primitives/moe_fused_compressed.hpp"


namespace ov {
namespace op {
namespace internal {
using MOEFusedCompressed  = ov::intel_gpu::op::MOEFusedCompressed ;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMOEFusedCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::MOEFusedCompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    OPENVINO_ASSERT(inputs.size() == 11, "Inputs count of MOEFusedCompressed should be 11");

    const std::string layerName = layer_type_name_ID(op);
    // auto& engine = p.get_engine();

    const cldnn::moe_fused_compressed moe(layerName, inputs, config);

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEFusedCompressed);

}  // namespace ov::intel_gpu
