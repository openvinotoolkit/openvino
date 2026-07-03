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

namespace {

// Bundle of pattern nodes captured by the shared callback. Each matcher instantiates
// its own copy of FC_COMPRESSED_WEIGHT_PATTERN (so the pattern node identities differ
// between the with-offsets and no-offsets matchers), then hands them to this helper.
struct GmmCompressedPatternNodes {
    std::shared_ptr<ov::Node> gmm_m;
    std::shared_ptr<ov::Node> mul_const_m;
    std::shared_ptr<ov::Node> decompressed_weights_m;
    std::shared_ptr<ov::Node> weights_const_m;
    std::shared_ptr<ov::Node> weights_param_m;
    std::shared_ptr<ov::Node> weights_reshape_m;
    std::shared_ptr<ov::Node> sub_const_m;
    std::shared_ptr<ov::Node> sub_with_convert_m;
    std::shared_ptr<ov::Node> sub_no_convert_m;
    std::shared_ptr<ov::Node> transpose_m;
    std::shared_ptr<ov::Node> transpose_const_m;
    std::shared_ptr<ov::Node> mul2_m;
    std::shared_ptr<ov::Node> mul2_const_m;
};

ov::matcher_pass_callback make_gmm_compressed_callback(const GmmCompressedPatternNodes& pn,
                                                      ov::pass::MatcherPass* self) {
    return [pn, self](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& gmm_m = pn.gmm_m;
        const auto& mul_const_m = pn.mul_const_m;
        const auto& decompressed_weights_m = pn.decompressed_weights_m;
        const auto& weights_const_m = pn.weights_const_m;
        const auto& weights_param_m = pn.weights_param_m;
        const auto& weights_reshape_m = pn.weights_reshape_m;
        const auto& sub_const_m = pn.sub_const_m;
        const auto& sub_with_convert_m = pn.sub_with_convert_m;
        const auto& sub_no_convert_m = pn.sub_no_convert_m;
        const auto& transpose_m = pn.transpose_m;
        const auto& transpose_const_m = pn.transpose_const_m;
        const auto& mul2_m = pn.mul2_m;
        const auto& mul2_const_m = pn.mul2_const_m;

        OPENVINO_ASSERT(pattern_map.count(gmm_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(decompressed_weights_m));

        auto gmm = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(pattern_map.at(gmm_m).get_node_shared_ptr());
        if (!gmm || self->transformation_callback(gmm)) {
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
        // v17::GroupedMatMul 2D x 3D expects weights [G, N, K] — rank 3 with G, N, K > 1 in practice.
        const bool is_weight_3d =
            std::count_if(weight_shape.begin(), weight_shape.end(), [](size_t d) { return d > 1; }) == 3;
        const bool grouped = scale_shape.size() == weight_shape.size() + 1;

        std::shared_ptr<ov::Node> weight_ptr = pattern_map.count(weights_const_m)
            ? pattern_map.at(weights_const_m).get_node_shared_ptr()
            : pattern_map.at(weights_param_m).get_node_shared_ptr();
        const bool weight_u8 = weight_ptr->get_element_type() == ov::element::u8 ||
                               weight_ptr->get_element_type() == ov::element::i8;

        auto reshape_const = [has_transpose, grouped, is_weight_3d](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            const ov::Shape current_shape = constant->get_shape();
            if (current_shape.size() <= 2)
                return constant;

            ov::Shape new_shape;
            if (current_shape.size() == 3) {
                if (is_weight_3d)
                    return constant;
                new_shape = (has_transpose || !grouped)
                                ? ov::Shape{current_shape[0] * current_shape[1], current_shape[2]}
                                : ov::Shape{current_shape[0], current_shape[1] * current_shape[2]};
            } else if (current_shape.size() == 4 && is_weight_3d) {
                new_shape = (has_transpose || !grouped)
                                ? ov::Shape{current_shape[0], current_shape[1] * current_shape[2], current_shape[3]}
                                : ov::Shape{current_shape[0], current_shape[1], current_shape[2] * current_shape[3]};
            } else if (current_shape.size() == 4 && !is_weight_3d) {
                new_shape =
                    (has_transpose || !grouped)
                        ? ov::Shape{current_shape[0] * current_shape[1] * current_shape[2], current_shape[3]}
                        : ov::Shape{current_shape[0] * current_shape[1], current_shape[2] * current_shape[3]};
            } else {
                OPENVINO_THROW("Unexpected constant shape rank ",
                               current_shape.size(),
                               " with is_weight_3d=",
                               is_weight_3d);
            }
            auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);
            ov::copy_weightless_cache_attr(constant, new_constant);
            return new_constant;
        };

        auto convert_const_to_u8 = [&](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            std::shared_ptr<ov::Node> result = nullptr;
            if (constant->get_element_type() == ov::element::u8)
                result = constant;
            else if (constant->get_element_type() == ov::element::u4)
                result = std::make_shared<ov::op::v0::Convert>(node, ov::element::u8);
            else if (weight_u8 && sub_with_convert && !constant->get_element_type().is_signed())
                result = std::make_shared<ov::op::v0::Convert>(node, ov::element::u8);
            else
                result = constant;

            ov::copy_weightless_cache_attr(node, result);
            return result;
        };

        const ov::Output<Node>& gmm_input_a = gmm->input(0).get_source_output();

        const auto& scale = reshape_const(pattern_map.at(mul_const_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        const bool with_zero_point =
            pattern_map.count(sub_no_convert_m) > 0 || pattern_map.count(sub_with_convert_m) > 0;
        if (with_zero_point) {
            optional_zero_point =
                convert_const_to_u8(reshape_const(pattern_map.at(sub_const_m).get_node_shared_ptr()));
        }

        std::shared_ptr<ov::Node> gmm_input_b = pattern_map.count(weights_const_m)
            ? reshape_const(pattern_map.at(weights_const_m).get_node_shared_ptr())
            : (pattern_map.count(weights_reshape_m)
                   ? pattern_map.at(weights_reshape_m).get_node_shared_ptr()
                   : pattern_map.at(weights_param_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> gmm_input_scale = scale;
        std::shared_ptr<ov::Node> gmm_input_zp = optional_zero_point;
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        if (gmm_input_b->get_output_partial_shape(0).size() != gmm_input_scale->get_shape().size()) {
            OPENVINO_ASSERT(!pattern_map.count(weights_const_m));
            ov::Shape weight_shape_final(gmm_input_scale->get_shape().size(), 1);
            for (size_t i = weight_shape.size() - 1, idx = gmm_input_scale->get_shape().size() - 1;; --i) {
                if (weight_shape[i] > 1) {
                    weight_shape_final[idx--] = weight_shape[i];
                }
                if (i == 0) {
                    break;
                }
            }
            if (has_transpose) {
                std::swap(weight_shape_final[weight_shape_final.size() - 1],
                          weight_shape_final[weight_shape_final.size() - 2]);
            }
            std::shared_ptr<ov::Node> weight_shape_const = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{weight_shape_final.size()}, weight_shape_final);
            gmm_input_b = std::make_shared<ov::op::v1::Reshape>(gmm_input_b, weight_shape_const, false);
            result_nodes.push_back(weight_shape_const);
            result_nodes.push_back(gmm_input_b);
        }

        if (has_transpose) {
            const auto& transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
            std::shared_ptr<ov::Node> transpose_const =
                pattern_map.at(transpose_const_m).get_node_shared_ptr();
            if (ov::shape_size(transpose_const->get_shape()) !=
                gmm_input_b->get_output_partial_shape(0).size()) {
                std::vector<int32_t> new_order(gmm_input_b->get_output_partial_shape(0).size());
                std::iota(new_order.begin(), new_order.end(), 0);
                std::swap(new_order[new_order.size() - 1], new_order[new_order.size() - 2]);
                transpose_const = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i32, ov::Shape{new_order.size()}, new_order);
            }

            gmm_input_b = transpose->clone_with_new_inputs({gmm_input_b->output(0), transpose_const});
            result_nodes.push_back(gmm_input_b);

            if (ov::shape_size(scale->output(0).get_shape()) > 1) {
                gmm_input_scale = transpose->clone_with_new_inputs({scale->output(0), transpose_const});
                result_nodes.push_back(gmm_input_scale);
            }

            if (with_zero_point && ov::shape_size(optional_zero_point->output(0).get_shape()) > 1) {
                gmm_input_zp =
                    transpose->clone_with_new_inputs({optional_zero_point->output(0), transpose_const});
                result_nodes.push_back(gmm_input_zp);
            }
        }

        if (pattern_map.count(mul2_m)) {
            auto mul2_op_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(mul2_const_m).get_node_shared_ptr());
            gmm_input_scale = ov::op::util::make_try_fold<ov::op::v1::Multiply>(gmm_input_scale, mul2_op_const);
        }

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
}

}  // namespace

ConvertGroupedMatMulWithOffsetsToCompressed::ConvertGroupedMatMulWithOffsetsToCompressed() {
    auto data_m = any_input();
    auto offsets_m = any_input();

    FC_COMPRESSED_WEIGHT_PATTERN

    auto gmm_m =
        wrap_type<ov::op::v17::GroupedMatMul>({data_m, compressed_weights_input_m, offsets_m});

    GmmCompressedPatternNodes pn{gmm_m,
                                 mul_const_m,
                                 decompressed_weights_m,
                                 weights_const_m,
                                 weights_param_m,
                                 weights_reshape_m,
                                 sub_const_m,
                                 sub_with_convert_m,
                                 sub_no_convert_m,
                                 transpose_m,
                                 transpose_const_m,
                                 mul2_m,
                                 mul2_const_m};

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_m,
                                                                "ConvertGroupedMatMulWithOffsetsToCompressed");
    this->register_matcher(matcher, make_gmm_compressed_callback(pn, this));
}

ConvertGroupedMatMulNoOffsetsToCompressed::ConvertGroupedMatMulNoOffsetsToCompressed() {
    auto data_m = any_input();

    FC_COMPRESSED_WEIGHT_PATTERN

    auto gmm_m = wrap_type<ov::op::v17::GroupedMatMul>({data_m, compressed_weights_input_m});

    GmmCompressedPatternNodes pn{gmm_m,
                                 mul_const_m,
                                 decompressed_weights_m,
                                 weights_const_m,
                                 weights_param_m,
                                 weights_reshape_m,
                                 sub_const_m,
                                 sub_with_convert_m,
                                 sub_no_convert_m,
                                 transpose_m,
                                 transpose_const_m,
                                 mul2_m,
                                 mul2_const_m};

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_m,
                                                                "ConvertGroupedMatMulNoOffsetsToCompressed");
    this->register_matcher(matcher, make_gmm_compressed_callback(pn, this));
}

ConvertGroupedMatMulToGroupedMatMulCompressed::ConvertGroupedMatMulToGroupedMatMulCompressed() {
    add_matcher<ConvertGroupedMatMulWithOffsetsToCompressed>();
    add_matcher<ConvertGroupedMatMulNoOffsetsToCompressed>();
}

}  // namespace ov::intel_gpu
