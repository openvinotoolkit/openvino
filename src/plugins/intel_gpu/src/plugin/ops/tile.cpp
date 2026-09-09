// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"
#include "openvino/op/constant.hpp"

#include <numeric>

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
        const auto input_pshape = op->get_input_partial_shape(0);

        if (input_pshape.rank().is_static() && repeats.size() > input_pshape.size()) {
            const auto rank_diff = repeats.size() - input_pshape.size();
            std::string reshapeName = layerName + "_reshape";

            if (op->is_dynamic() || p.use_new_shape_infer()) {
                auto output_pshape = input_pshape;
                output_pshape.insert(output_pshape.begin(), rank_diff, 1);

                std::vector<int64_t> axes(rank_diff);
                std::iota(axes.begin(), axes.end(), 0);

                auto reshapePrim = cldnn::reshape(reshapeName,
                                                  inputs[0],
                                                  false,
                                                  axes,
                                                  output_pshape,
                                                  cldnn::reshape::reshape_mode::unsqueeze);
                p.add_primitive(*op, reshapePrim);
            } else {
                auto inputDims = op->get_input_shape(0);
                inputDims.insert(inputDims.begin(), rank_diff, 1);

                auto targetShape = tensor_from_dims(inputDims);
                auto reshapePrim = cldnn::reshape(reshapeName, inputs[0], targetShape);
                p.add_primitive(*op, reshapePrim);
            }

            inputs[0] = cldnn::input_info(reshapeName);
        } else if (!op->is_dynamic() && !p.use_new_shape_infer()) {
            size_t rank = op->get_input_shape(0).size();
            int64_t defaultSize = 1;
            for (size_t i = repeats.size(); i < rank; ++i) {
                repeats.insert(repeats.begin(), defaultSize);
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
