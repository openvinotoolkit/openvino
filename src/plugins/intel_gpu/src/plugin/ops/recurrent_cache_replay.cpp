// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "intel_gpu/op/recurrent_cache_replay.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/recurrent_cache_replay.hpp"

namespace ov::op::internal {
using ov::intel_gpu::op::RecurrentConvCacheReplayUpdate;
using ov::intel_gpu::op::RecurrentGatedDeltaNetReplayUpdate;
} // namespace ov::op::internal

namespace ov::intel_gpu {
static void
CreateRecurrentConvCacheReplayUpdateOp(ProgramBuilder& builder,
                                       const std::shared_ptr<ov::op::internal::RecurrentConvCacheReplayUpdate>& op) {
    validate_inputs_count(op, {6});
    builder.add_primitive(
        *op, cldnn::recurrent_cache_replay(layer_type_name_ID(op), builder.GetInputInfo(op), op->get_history()));
}
static void CreateRecurrentGatedDeltaNetReplayUpdateOp(
    ProgramBuilder& builder, const std::shared_ptr<ov::op::internal::RecurrentGatedDeltaNetReplayUpdate>& op) {
    validate_inputs_count(op, {12});
    builder.add_primitive(*op,
                          cldnn::recurrent_cache_replay(layer_type_name_ID(op), builder.GetInputInfo(op), 0,
                                                        op->get_fuse_qk_l2norm(), op->get_q_l2_norm_eps(),
                                                        op->get_k_l2_norm_eps(), op->get_head_mapping() == "grouped"));
}
REGISTER_FACTORY_IMPL(internal, RecurrentConvCacheReplayUpdate);
REGISTER_FACTORY_IMPL(internal, RecurrentGatedDeltaNetReplayUpdate);
} // namespace ov::intel_gpu
