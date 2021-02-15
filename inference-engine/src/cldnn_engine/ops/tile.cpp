// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/tile.hpp"

#include "api/tile.hpp"

namespace CLDNNPlugin {

void CreateTileOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tile>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto tilePrim = cldnn::tile(layerName,
                                inputPrimitives[0],
                                CldnnTensorFromIEDims(op->get_output_shape(0)));

    p.AddPrimitive(tilePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, Tile);

}  // namespace CLDNNPlugin
