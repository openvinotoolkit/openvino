// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/scatter_nd_update.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/scatter_nd_update.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateScatterNDUpdateOp(Program& p, const std::shared_ptr<ngraph::op::v3::ScatterNDUpdate>& op) {
    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto indices_rank = op->get_input_shape(1).size();

    auto primitive = cldnn::scatter_nd_update(layerName,
                                              inputPrimitives[0],
                                              inputPrimitives[1],
                                              inputPrimitives[2],
                                              indices_rank,
                                              op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, ScatterNDUpdate);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
