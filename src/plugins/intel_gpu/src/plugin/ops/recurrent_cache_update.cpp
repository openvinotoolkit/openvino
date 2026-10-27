// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "openvino/op/recurrent_cache_update.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/recurrent_cache_update.hpp"
namespace ov::intel_gpu {
static void CreateRecurrentConvCacheUpdateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::RecurrentConvCacheUpdate>& op) {
    validate_inputs_count(op, {8, 9});
    p.add_primitive(*op, cldnn::recurrent_cache_update(layer_type_name_ID(op), p.GetInputInfo(op), op->get_history()));
}
static void CreateRecurrentGatedDeltaNetUpdateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::RecurrentGatedDeltaNetUpdate>& op) {
    validate_inputs_count(op, {12});
    p.add_primitive(*op,
                    cldnn::recurrent_cache_update(layer_type_name_ID(op),
                                                  p.GetInputInfo(op),
                                                  0,
                                                  op->get_fuse_qk_l2norm(),
                                                  op->get_q_l2_norm_eps(),
                                                  op->get_k_l2_norm_eps(),
                                                  op->get_snapshot_count()));
}
REGISTER_FACTORY_IMPL(internal, RecurrentConvCacheUpdate);
REGISTER_FACTORY_IMPL(internal, RecurrentGatedDeltaNetUpdate);
}  // namespace ov::intel_gpu
