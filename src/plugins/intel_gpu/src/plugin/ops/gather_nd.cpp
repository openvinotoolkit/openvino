// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/gather_nd.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/gather_nd.hpp"

namespace ov::intel_gpu {

static void CreateGatherNDOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::GatherND>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_rank = static_cast<const uint8_t>(op->get_input_partial_shape(0).size());
    auto indices_rank = static_cast<const uint8_t>(op->get_input_partial_shape(1).size());
    auto batch_dims = static_cast<const uint8_t>(op->get_batch_dims());

    auto primitive = cldnn::gather_nd(layerName,
                                      inputs[0],
                                      inputs[1],
                                      input_rank,
                                      indices_rank,
                                      batch_dims);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v5, GatherND);

static void CreateGatherNDOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::GatherND>& op) {
    validate_inputs_count(op, { 2 });
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_rank = static_cast<const uint8_t>(op->get_input_partial_shape(0).size());
    auto indices_rank = static_cast<const uint8_t>(op->get_input_partial_shape(1).size());
    auto batch_dims = static_cast<const uint8_t>(op->get_batch_dims());

    auto primitive = cldnn::gather_nd(layerName,
                                      inputs[0],
                                      inputs[1],
                                      input_rank,
                                      indices_rank,
                                      batch_dims,
                                      false);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v8, GatherND);

}  // namespace ov::intel_gpu
