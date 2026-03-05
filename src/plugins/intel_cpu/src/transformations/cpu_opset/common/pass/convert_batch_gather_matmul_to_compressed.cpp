// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_batch_gather_matmul_to_compressed.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

ov::intel_cpu::ConvertBatchGatherMatmulToBatchGatherMatmulCompressed::
    ConvertBatchGatherMatmulToBatchGatherMatmulCompressed(
        const std::vector<ov::element::Type>& supported_activation_types,
        const std::vector<ov::element::Type>& supported_weights_types,
        const SupportsPredicate& supports_config,
        bool convert_u4zp_to_u8) {
    using namespace ov::pass::pattern;

    auto activation = any_input(type_matches_any(supported_activation_types));
    auto weights_block =
        std::make_shared<ov::pass::pattern::op::CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{3});
    auto indices = any_input();
    auto bias = any_input();
    auto batch_gather_matmul = wrap_type<BatchGatherMatmul>({activation, weights_block, indices, bias});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto bgm = ov::as_type_ptr<BatchGatherMatmul>(pattern_map.at(batch_gather_matmul).get_node_shared_ptr());
        if (!bgm || transformation_callback(bgm)) {
            return false;
        }

        bool has_transpose = weights_block->get_anchor("transpose", pattern_map).has_value();
        const auto& weights_shape = bgm->get_input_shape(1);
        bool batched_weights = weights_shape.size() == 3 && weights_shape[0] > 1;
        auto scale_shape = weights_block->get_anchor("mul_const", pattern_map).value().get_shape();
        bool grouped = std::count_if(scale_shape.begin(), scale_shape.end(), [](size_t d) {
                           return d > 1;
                       }) > (batched_weights ? 2 : 1);

        ov::NodeVector result_nodes;
        const auto [bgm_input_b, bgm_input_scale, bgm_input_zp] =
            ov::pass::ConvertFullyConnectedToFullyConnectedCompressed::process_compressed_weights(weights_block,
                                                                                                  pattern_map,
                                                                                                  convert_u4zp_to_u8,
                                                                                                  has_transpose,
                                                                                                  grouped,
                                                                                                  batched_weights,
                                                                                                  result_nodes);

        auto new_bgm = std::make_shared<BatchGatherMatmulCompressed>(pattern_map.at(activation),
                                                                     bgm_input_b,
                                                                     pattern_map.at(indices),
                                                                     pattern_map.at(bias),
                                                                     bgm_input_scale,
                                                                     bgm_input_zp);

        const size_t IC = *(weights_shape.rbegin());
        const size_t OC = *(weights_shape.rbegin() + 1);
        size_t G = 1;
        if (grouped) {
            G = has_transpose ? *(scale_shape.rbegin() + 2) : *(scale_shape.rbegin() + 1);
        }
        if (supports_config && !supports_config(new_bgm, IC, OC, G)) {
            return false;
        }

        result_nodes.push_back(new_bgm);
        new_bgm->set_friendly_name(bgm->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(bgm, new_bgm);
        return true;
    };

    auto m = std::make_shared<Matcher>(batch_gather_matmul, "ConvertBatchGatherMatmulToBatchGatherMatmulCompressed");
    this->register_matcher(m, callback);
}