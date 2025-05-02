// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_qkv_proj.hpp"

#include <memory>
#include <optional>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;

namespace {

/**
 * @brief L2 normalization pattern:
 * Power(x, 2) → ReduceSum → Sqrt → Divide → Multiply
 */
Output<Node> match_l2_norm(const Output<Node>& input) {
    auto pow_const = pass::pattern::wrap_type<v0::Constant>();
    auto pow = pass::pattern::wrap_type<v1::Power>({input, pow_const});

    auto reduce_axes = pass::pattern::wrap_type<v0::Constant>();
    auto reduce = pass::pattern::wrap_type<v1::ReduceSum>({pow, reduce_axes});

    auto sqrt = pass::pattern::wrap_type<v0::Sqrt>({reduce});
    auto div = pass::pattern::wrap_type<v1::Divide>({input, sqrt});

    auto scale_const = pass::pattern::wrap_type<v0::Constant>();
    auto norm = pass::pattern::wrap_type<v1::Multiply>({div, scale_const});

    return norm;
}

/**
 * @brief Extracts a weight node usable for Concat (either Constant or quantized path).
 * Returns: (node to concat, original Constant for shape info)
 */
std::optional<std::tuple<Output<Node>, std::shared_ptr<v0::Constant>>> get_weight_or_dequant(const Output<Node>& node) {
    if (const auto& c = ov::as_type_ptr<v0::Constant>(node.get_node_shared_ptr()))
        return {{c, c}};

    auto mul = ov::as_type_ptr<v1::Multiply>(node.get_node_shared_ptr());
    if (!mul)
        return std::nullopt;

    auto sub = ov::as_type_ptr<v1::Subtract>(mul->input_value(0).get_node_shared_ptr());
    auto scale = ov::as_type_ptr<v0::Constant>(mul->input_value(1).get_node_shared_ptr());
    if (!sub || !scale)
        return std::nullopt;

    auto conv = ov::as_type_ptr<v0::Convert>(sub->input_value(0).get_node_shared_ptr());
    auto zp_conv = ov::as_type_ptr<v0::Convert>(sub->input_value(1).get_node_shared_ptr());
    if (!conv || !zp_conv)
        return std::nullopt;

    auto w_const = ov::as_type_ptr<v0::Constant>(conv->input_value(0).get_node_shared_ptr());
    auto zp_const = ov::as_type_ptr<v0::Constant>(zp_conv->input_value(0).get_node_shared_ptr());
    if (!w_const || !zp_const)
        return std::nullopt;

    return {{mul, w_const}};
}

}  // namespace

ov::pass::PackQKVProj::PackQKVProj() {
    MATCHER_SCOPE(L2SharedInputMatMulFusion);

    auto input = pattern::any_input();
    auto norm_output = match_l2_norm(input);

    // Match one representative MatMul that uses this normed output
    auto weight = pattern::any_input();
    auto matmul = pattern::wrap_type<v0::MatMul>({norm_output, weight});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto norm_node = pm.at(norm_output.get_node_shared_ptr()).get_node_shared_ptr();

        // Gather all MatMul-s with this normalized input
        ov::NodeVector matmuls;
        for (const auto& input : norm_node->output(0).get_target_inputs()) {
            if (auto mm = as_type<v0::MatMul>(input.get_node())) {
                if (mm->input_value(0).get_node() == norm_node.get())
                    matmuls.push_back(mm->shared_from_this());
            }
        }

        if (matmuls.size() < 2)
            return false;

        OutputVector fused_weights;
        std::vector<int64_t> split_sizes;

        for (auto& mm : matmuls) {
            auto result = get_weight_or_dequant(mm->input_value(1));
            if (!result)
                return false;
            auto [dequant, orig_const] = *result;
            fused_weights.push_back(dequant);
            split_sizes.push_back(orig_const->get_shape().at(1));
        }

        // Combine all weights into one MatMul
        // TODO: Should we leave Concat op?
        auto concat_weights = ov::op::util::make_try_fold<v0::Concat>(fused_weights, 1);
        auto fused_mm = std::make_shared<v0::MatMul>(norm_node, concat_weights);

        // Split back to outputs
        auto axis = v0::Constant::create(element::i64, Shape{}, {1});
        auto sizes = v0::Constant::create(element::i64, Shape{split_sizes.size()}, split_sizes);
        auto split = std::make_shared<v1::VariadicSplit>(fused_mm, axis, sizes);

        for (size_t i = 0; i < matmuls.size(); ++i) {
            if (!replace_output_update_name(matmuls[i]->output(0), split->output(i)))
                return false;
        }
        copy_runtime_info({matmuls}, {concat_weights, fused_mm, split});

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(matmul, matcher_name);
    register_matcher(matcher, callback);
}
