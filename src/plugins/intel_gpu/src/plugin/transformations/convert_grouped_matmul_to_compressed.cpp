// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_grouped_matmul_to_compressed.hpp"

#include <algorithm>
#include <memory>
#include <numeric>

#include "intel_gpu/op/grouped_matmul_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"


#include "compressed_weights_pattern.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

ConvertGroupedMatMulToGroupedMatMulCompressed::ConvertGroupedMatMulToGroupedMatMulCompressed() {
    auto data_m = any_input();
    auto offsets_m = any_input();

    FC_COMPRESSED_WEIGHT_PATTERN

    // v17::GroupedMatMul has two legal input arities. Match both with a single Or root so
    // one MatcherPass covers 2D x 3D (with offsets) and 3D x 3D (no offsets).
    auto gmm_no_offsets_m =
        wrap_type<ov::op::v17::GroupedMatMul>({data_m, compressed_weights_input_m});
    auto gmm_with_offsets_m =
        wrap_type<ov::op::v17::GroupedMatMul>({data_m, compressed_weights_input_m, offsets_m});
    auto gmm_m = std::make_shared<ov::pass::pattern::op::Or>(
        ov::OutputVector{gmm_no_offsets_m, gmm_with_offsets_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(decompressed_weights_m));

        auto gmm = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }

        const auto& a_pshape = gmm->get_input_partial_shape(0);
        if (a_pshape.rank().is_dynamic()) {
            return false;
        }
        const auto a_rank = a_pshape.size();
        const bool with_offsets = (gmm->get_input_size() == 3);
        if (with_offsets) {
            // 2D x 3D form must carry offsets and a rank-2 activation.
            if (a_rank != 2) {
                return false;
            }
        } else {
            // 3D x 3D form has no offsets and rank-3 activation.
            if (a_rank != 3 || gmm->get_input_size() != 2) {
                return false;
            }
        }

        const bool has_transpose = pattern_map.count(transpose_m) > 0;
        const auto scale_shape = pattern_map.at(mul_const_m).get_shape();
        const bool sub_with_convert = pattern_map.count(sub_with_convert_m) > 0;
        const auto weight_shape = gmm->get_input_shape(1);
        // GroupedMatMul-17 always has rank-3 mat_b ([G, N, K]); the callback runs only
        // after that pattern matched, so treat it as an invariant rather than a runtime bool.
        OPENVINO_ASSERT(weight_shape.size() == 3,
                        "GroupedMatMul mat_b must be rank 3, got rank ", weight_shape.size());

        // If the scale/zero-point has one extra dimension (rank 4), the K axis has been split
        // into sub-groups that each share a scale
        // e.g. shape [G, N, K/group_size, 1] or [G, K/group_size, group_size, N].
        const bool grouped = scale_shape.size() == weight_shape.size() + 1;

        std::shared_ptr<ov::Node> weight_ptr = pattern_map.count(weights_const_m)
            ? pattern_map.at(weights_const_m).get_node_shared_ptr()
            : pattern_map.at(weights_param_m).get_node_shared_ptr();
        const bool weight_u8 = weight_ptr->get_element_type() == ov::element::u8 ||
                               weight_ptr->get_element_type() == ov::element::i8;

        auto reshape_const_to_3d = [has_transpose, grouped](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            const ov::Shape current_shape = constant->get_shape();
            // Rank <= 3 already matches the rank-3 weight; only rank-4 grouped scales/zp
            // need flattening down to rank 3.
            if (current_shape.size() <= 3)
                return constant;

            OPENVINO_ASSERT(current_shape.size() == 4,
                            "Unexpected decompression constant rank ", current_shape.size());
            const ov::Shape new_shape =
                (has_transpose || !grouped)
                    ? ov::Shape{current_shape[0], current_shape[1] * current_shape[2], current_shape[3]}
                    : ov::Shape{current_shape[0], current_shape[1], current_shape[2] * current_shape[3]};
            auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);
            ov::copy_weightless_cache_attr(constant, new_constant);
            return new_constant;
        };

        auto convert_const_to_u8 = [&](std::shared_ptr<ov::Node> node) -> std::shared_ptr<ov::Node> {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            const auto elem_type = constant->get_element_type();
            // Promote to u8 for:
            //   - u4 zero-points (always widened),
            //   - other sub-byte unsigned zero-points when the weight is (u|i)8 and the
            //     subtract branch already inserts a Convert (matches FC-compressed behavior).
            // u8 and signed types are passed through unchanged.
            const bool promote_to_u8 =
                elem_type == ov::element::u4 ||
                (elem_type != ov::element::u8 && !elem_type.is_signed() && weight_u8 && sub_with_convert);
            if (!promote_to_u8)
                return node;

            auto converted = std::make_shared<ov::op::v0::Convert>(node, ov::element::u8);
            ov::copy_weightless_cache_attr(node, converted);
            return converted;
        };

        const ov::Output<Node>& gmm_input_a = gmm->input(0).get_source_output();

        const auto& scale = reshape_const_to_3d(pattern_map.at(mul_const_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        const bool with_zero_point =
            pattern_map.count(sub_no_convert_m) > 0 || pattern_map.count(sub_with_convert_m) > 0;
        if (with_zero_point) {
            optional_zero_point =
                convert_const_to_u8(reshape_const_to_3d(pattern_map.at(sub_const_m).get_node_shared_ptr()));
        }

        std::shared_ptr<ov::Node> gmm_input_b = pattern_map.count(weights_const_m)
            ? reshape_const_to_3d(pattern_map.at(weights_const_m).get_node_shared_ptr())
            : (pattern_map.count(weights_reshape_m)
                   ? pattern_map.at(weights_reshape_m).get_node_shared_ptr()
                   : pattern_map.at(weights_param_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> gmm_input_scale = scale;
        std::shared_ptr<ov::Node> gmm_input_zp = optional_zero_point;
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        // 3 conditions below are from convert_fc_to_compressed. It was not observed from grouped matmul compressed, yet.
        OPENVINO_ASSERT(gmm_input_b->get_output_partial_shape(0).size() == gmm_input_scale->get_shape().size(),
                        "GroupedMatMulCompressed expects the decompression scale to have the same rank as mat_b, "
                        "got scale rank ",
                        gmm_input_scale->get_shape().size(),
                        " and mat_b rank ",
                        gmm_input_b->get_output_partial_shape(0).size());

        OPENVINO_ASSERT(!has_transpose,
                        "GroupedMatMulCompressed does not support transposed weights, but the pattern matched a transpose.");

        OPENVINO_ASSERT(pattern_map.count(mul2_m) == 0,
                        "GroupedMatMulCompressed does not support a second multiply after the decompression scale, "
                        "but the pattern matched a second multiply.");

        std::shared_ptr<ov::Node> new_gmm = nullptr;
        if (with_offsets) {
            const ov::Output<Node>& gmm_input_offsets = gmm->input(2).get_source_output();
            if (with_zero_point) {
                new_gmm = std::make_shared<op::GroupedMatMulCompressed>(gmm_input_a,
                                                                        gmm_input_b,
                                                                        gmm_input_offsets,
                                                                        gmm_input_scale,
                                                                        gmm_input_zp);
            } else {
                new_gmm = std::make_shared<op::GroupedMatMulCompressed>(gmm_input_a,
                                                                        gmm_input_b,
                                                                        gmm_input_offsets,
                                                                        gmm_input_scale);
            }
        } else {
            if (with_zero_point) {
                new_gmm = op::GroupedMatMulCompressed::make_3d(gmm_input_a,
                                                               gmm_input_b,
                                                               gmm_input_scale,
                                                               gmm_input_zp);
            } else {
                new_gmm = op::GroupedMatMulCompressed::make_3d(gmm_input_a,
                                                               gmm_input_b,
                                                               gmm_input_scale);
            }
        }

        result_nodes.push_back(new_gmm);
        new_gmm->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(gmm, new_gmm);
        return true;
    };

    auto matcher =
        std::make_shared<ov::pass::pattern::Matcher>(gmm_m, "ConvertGroupedMatMulToGroupedMatMulCompressed");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
