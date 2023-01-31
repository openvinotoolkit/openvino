// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/tile.hpp"

#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTileOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tile>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto repeatsNode = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!repeatsNode)
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() <<
                                                        " (" << op->get_type_name() << ")";
    std::vector<int64_t> repeats = repeatsNode->cast_vector<int64_t>();

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

    auto tilePrim = cldnn::tile(layerName,
                                inputs[0],
                                repeats);

    p.add_primitive(*op, tilePrim);
}

REGISTER_FACTORY_IMPL(v0, Tile);

}  // namespace intel_gpu
}  // namespace ov
