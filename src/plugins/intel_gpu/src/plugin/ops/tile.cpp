// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov::intel_gpu {

static void CreateTileOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Tile>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    if (auto repeats_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1))) {
        std::vector<int64_t> repeats = repeats_const->cast_vector<int64_t>();

        // TODO: Remove code below once new shape infer is enabled
        if (!op->is_dynamic() && !p.use_new_shape_infer()) {
            size_t rank = op->get_input_shape(0).size();
            int64_t defaultSize = 1;
            for (size_t i = repeats.size(); i < rank; ++i) {
                repeats.insert(repeats.begin(), defaultSize);
            }

            if (repeats.size() > rank) {
                std::string reshapeName = layerName + "_reshape";
                auto inputDims = op->get_input_shape(0);

                // Extend input dimensions to the same size as repeats dimensions by prepending ones
                inputDims.insert(inputDims.begin(), repeats.size() - rank, defaultSize);

                auto targetShape = tensor_from_dims(inputDims);

                auto reshapePrim = cldnn::reshape(reshapeName, inputs[0], targetShape);

                p.add_primitive(*op, reshapePrim);

                inputs[0] = cldnn::input_info(reshapeName);
            }
        }

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

}  // namespace ov::intel_gpu
