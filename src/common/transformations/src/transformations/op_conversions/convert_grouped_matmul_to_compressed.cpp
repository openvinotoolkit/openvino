// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_compressed.hpp"

#include <cstddef>
#include <memory>
#include <set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

ov::pass::ConvertGroupedMatMulToGroupedMatMulCompressed::ConvertGroupedMatMulToGroupedMatMulCompressed(
    const std::vector<ov::element::Type>& supported_weights_types) {
    using namespace ov::pass::pattern;
    using ov::op::internal::GroupedMatMulCompressed;

    auto data_2d_m = any_input(rank_equals(2));
    auto data_3d_m = any_input(rank_equals(3));
    auto offsets_m = any_input();
    auto weights_block =
        std::make_shared<ov::pass::pattern::op::CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{3});

    // v17::GroupedMatMul has two legal input arities. Match both with a single Or root so
    // one MatcherPass covers 2D x 3D (with offsets) and 3D x 3D (no offsets). Rank is
    // enforced by the pattern predicates above so the callback doesn't need to re-check it.
    auto gmm_no_offsets_m = wrap_type<ov::op::v17::GroupedMatMul>({data_3d_m, weights_block});
    auto gmm_with_offsets_m = wrap_type<ov::op::v17::GroupedMatMul>({data_2d_m, weights_block, offsets_m});
    auto gmm_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{gmm_no_offsets_m, gmm_with_offsets_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gmm = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }

        // If the CompressedWeightsBlock matched a Transpose on the weights branch, the
        // original graph is producing [G, N, K] mat_b from a differently-laid-out constant.
        // The shared helper bakes that Transpose into the weights/scale/zp constants when
        // `has_transpose=true`, so the emitted GroupedMatMulCompressed still sees [G, N, K].
        const bool has_transpose = weights_block->get_anchor("transpose", pattern_map).has_value();

        // Which arm of the Or matched — 2D x 3D (with offsets) or 3D x 3D (no offsets).
        const bool with_offsets = pattern_map.count(data_2d_m) > 0;
        const auto& data_value = with_offsets ? pattern_map.at(data_2d_m) : pattern_map.at(data_3d_m);

        const auto& weights_shape = gmm->get_input_shape(1);
        // GroupedMatMul-17 always has rank-3 mat_b ([G, N, K])
        OPENVINO_ASSERT(weights_shape.size() == 3,
                        "GroupedMatMul mat_b must be rank 3, got rank ",
                        weights_shape.size());

        // If the scale/zero-point has one extra dimension (rank 4), the K axis has been split
        // into sub-groups that each share a scale, e.g. shape [G, N, K/group_size, group_size].
        const auto scale_shape = weights_block->get_anchor("mul_const", pattern_map).value().get_shape();
        const bool grouped = scale_shape.size() == weights_shape.size() + 1;

        ov::NodeVector result_nodes;
        // `batched_weights=true` selects a final weights rank of 3 in the shared helper, which
        // matches the rank-3 mat_b of GroupedMatMul.
        const auto [gmm_input_b, gmm_input_scale, gmm_input_zp] =
            ov::pass::ConvertFullyConnectedToFullyConnectedCompressed::process_compressed_weights(
                weights_block,
                pattern_map,
                /*convert_u4zp_to_u8=*/true,
                has_transpose,
                grouped,
                /*batched_weights=*/true,
                result_nodes);

        const bool with_zero_point = weights_block->get_anchor("sub_no_convert", pattern_map).has_value() ||
                                     weights_block->get_anchor("sub_with_convert", pattern_map).has_value();

        std::shared_ptr<ov::Node> new_gmm;
        if (with_offsets) {
            const auto& gmm_input_offsets = gmm->input_value(2);
            if (with_zero_point) {
                new_gmm = std::make_shared<GroupedMatMulCompressed>(data_value,
                                                                    gmm_input_b,
                                                                    gmm_input_offsets,
                                                                    gmm_input_scale,
                                                                    gmm_input_zp);
            } else {
                new_gmm = std::make_shared<GroupedMatMulCompressed>(data_value,
                                                                    gmm_input_b,
                                                                    gmm_input_offsets,
                                                                    gmm_input_scale);
            }
        } else {
            if (with_zero_point) {
                new_gmm = GroupedMatMulCompressed::make_3d(data_value, gmm_input_b, gmm_input_scale, gmm_input_zp);
            } else {
                new_gmm = GroupedMatMulCompressed::make_3d(data_value, gmm_input_b, gmm_input_scale);
            }
        }

        result_nodes.push_back(new_gmm);
        new_gmm->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(gmm, new_gmm);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_m, "ConvertGroupedMatMulToGroupedMatMulCompressed");
    this->register_matcher(matcher, callback);
}
