// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/rotary_positional_embeddings.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "plugin/transformations/fuse_rms_rope.hpp"

namespace ov {
namespace op {
namespace internal {
using RoPE = ov::op::internal::RoPE;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateRoPEOp(ProgramBuilder& p, const std::shared_ptr<op::internal::RoPE>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();

    size_t gather_rank = 0;
    if (config.gather_position_arg_id > 0) {
        gather_rank = op->get_input_partial_shape(config.gather_position_arg_id).size();
    }

    OPENVINO_ASSERT(!config.is_interleaved || !config.output_trans0213, "[GPU] Unsupported ROPE parameters");

    const auto& rt_info = op->get_rt_info();
    const auto fused_rms = rt_info.find(fuse_rms_rope_epsilon_key);
    const bool fuse_rms_norm = fused_rms != rt_info.end();
    float rms_epsilon = 0.0f;
    if (fuse_rms_norm) {
        OPENVINO_ASSERT(inputs.size() == 4 && gather_rank == 0, "[GPU] Invalid fused RMSNorm and RoPE inputs");
        rms_epsilon = fused_rms->second.as<float>();
    }

    auto rope = cldnn::rope(layer_type_name_ID(op),
                            inputs,
                            config,
                            gather_rank,
                            fuse_rms_norm,
                            rms_epsilon);

    p.add_primitive(*op, rope);
}

REGISTER_FACTORY_IMPL(internal, RoPE);

}  // namespace ov::intel_gpu
