// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/broadcast.hpp"

#include "api/broadcast.hpp"

namespace CLDNNPlugin {

static void CreateCommonBroadcastOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {2, 3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto broadcastPrim = cldnn::broadcast(layerName,
                                          inputPrimitives[0],
                                          CldnnTensorFromIEDims(op->get_output_shape(0)));

    p.AddPrimitive(broadcastPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateBroadcastOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    CreateCommonBroadcastOp(p, node);
}

REGISTER_FACTORY_IMPL(v1, Broadcast);
REGISTER_FACTORY_IMPL(v3, Broadcast);

}  // namespace CLDNNPlugin
