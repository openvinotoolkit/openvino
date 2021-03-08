// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/reorg_yolo.hpp"

#include "api/reorg_yolo.hpp"

namespace CLDNNPlugin {

void CreateReorgYoloOp(Program& p, const std::shared_ptr<ngraph::op::v0::ReorgYolo>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t stride = op->get_strides()[0];

    auto reorgPrim = cldnn::reorg_yolo(layerName,
                                       inputPrimitives[0],
                                       stride);

    p.AddPrimitive(reorgPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ReorgYolo);

}  // namespace CLDNNPlugin
