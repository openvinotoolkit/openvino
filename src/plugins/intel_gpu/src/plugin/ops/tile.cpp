// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/tile.hpp"

#include "intel_gpu/primitives/tile.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateTileOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tile>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto tilePrim = cldnn::tile(layerName,
                                inputPrimitives[0],
                                tensor_from_dims(op->get_output_shape(0)),
                                op->get_friendly_name());

    p.AddPrimitive(tilePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, Tile);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
