// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/tile.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTileOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Tile>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    if (auto repeats_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1))) {
        std::vector<int64_t> repeats = repeats_const->cast_vector<int64_t>();

        p.add_primitive(*op, cldnn::tile(layerName,
                                         inputs[0],
                                         repeats));
    } else {
        p.add_primitive(*op, cldnn::tile(layerName,
                                         inputs[0],
                                         inputs[1]));
    }
}

REGISTER_FACTORY_IMPL(v0, Tile);

}  // namespace intel_gpu
}  // namespace ov
