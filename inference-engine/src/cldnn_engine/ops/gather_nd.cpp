// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/constant.hpp"

#include "api/gather_nd.hpp"

namespace CLDNNPlugin {

void CreateGatherNDOp(Program& p, const std::shared_ptr<ngraph::op::v5::GatherND>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t indices_rank = static_cast<int32_t>(op->get_input_shape(1).size());

    auto batch_dims = op->get_batch_dims();

    auto primitive = cldnn::gather_nd(layerName,
                                           inputPrimitives[0],
                                           inputPrimitives[1],
                                           indices_rank,
                                           batch_dims);

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v5, GatherND);

}  // namespace CLDNNPlugin
