// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/gather_nd.hpp"

namespace ov {
namespace intel_gpu {

static void CreateGatherNDOp(Program& p, const std::shared_ptr<ngraph::op::v5::GatherND>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t input_rank = static_cast<int32_t>(op->get_input_shape(0).size());
    int32_t indices_rank = static_cast<int32_t>(op->get_input_shape(1).size());

    auto batch_dims = op->get_batch_dims();

    auto primitive = cldnn::gather_nd(layerName,
                                      inputPrimitives[0],
                                      inputPrimitives[1],
                                      input_rank,
                                      indices_rank,
                                      batch_dims,
                                      true,
                                      op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v5, GatherND);

static void CreateGatherNDOp(Program& p, const std::shared_ptr<ngraph::op::v8::GatherND>& op) {
    p.ValidateInputs(op, { 2 });
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t input_rank = static_cast<int32_t>(op->get_input_shape(0).size());
    int32_t indices_rank = static_cast<int32_t>(op->get_input_shape(1).size());

    auto batch_dims = op->get_batch_dims();

    auto primitive = cldnn::gather_nd(layerName,
                                      inputPrimitives[0],
                                      inputPrimitives[1],
                                      input_rank,
                                      indices_rank,
                                      batch_dims,
                                      false,
                                      op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v8, GatherND);

}  // namespace intel_gpu
}  // namespace ov
