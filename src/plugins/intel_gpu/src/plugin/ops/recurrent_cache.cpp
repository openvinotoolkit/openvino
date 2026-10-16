// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/recurrent_cache.hpp"
namespace ov::intel_gpu {
static void CreateRecurrentStateGatherOp(ProgramBuilder &p,
                                         const std::shared_ptr<ov::op::internal::RecurrentStateGather> &op) {
    validate_inputs_count(op, {6});
    p.add_primitive(*op, cldnn::recurrent_cache(layer_type_name_ID(op), p.GetInputInfo(op), 0, op->get_key_size(),
                                                op->get_value_size()));
}
static void CreateRecurrentConvCacheOp(ProgramBuilder &p,
                                       const std::shared_ptr<ov::op::internal::RecurrentConvCache> &op) {
    validate_inputs_count(op, {8});
    p.add_primitive(*op, cldnn::recurrent_cache(layer_type_name_ID(op), p.GetInputInfo(op), op->get_history()));
}
static void CreateRecurrentStateUpdateOp(ProgramBuilder &p,
                                         const std::shared_ptr<ov::op::internal::RecurrentStateUpdate> &op) {
    validate_inputs_count(op, {3});
    p.add_primitive(*op, cldnn::recurrent_cache(layer_type_name_ID(op), p.GetInputInfo(op), -1, op->get_key_size(),
                                                op->get_value_size()));
}
REGISTER_FACTORY_IMPL(internal, RecurrentStateUpdate);
REGISTER_FACTORY_IMPL(internal, RecurrentStateGather);
REGISTER_FACTORY_IMPL(internal, RecurrentConvCache);
} // namespace ov::intel_gpu
