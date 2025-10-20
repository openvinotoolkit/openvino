// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/moe_compressed.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_compressed.hpp"

namespace ov::intel_gpu {

static void CreateMOECompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOECompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    OPENVINO_ASSERT(inputs.size() == 12, "Inputs count of MOE should be 12");

    const std::string layerName = layer_type_name_ID(op);
    // auto& engine = p.get_engine();

    const cldnn::moe_compressed moe(layerName, inputs, config);

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOECompressed);

}  // namespace ov::intel_gpu
