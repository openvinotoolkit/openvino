// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/scatter_nd_update.hpp"
#include "ngraph/op/constant.hpp"

#include "api/scatter_nd_update.hpp"

namespace CLDNNPlugin {

void CreateScatterNDUpdateOp(Program& p, const std::shared_ptr<ngraph::op::v3::ScatterNDUpdate>& op) {
    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto indices_rank = op->get_input_shape(1).size();

    auto primitive = cldnn::scatter_nd_update(layerName,
                                           inputPrimitives[0],
                                           inputPrimitives[1],
                                           inputPrimitives[2],
                                           indices_rank);

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, ScatterNDUpdate);

}  // namespace CLDNNPlugin