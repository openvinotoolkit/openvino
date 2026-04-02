// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/paged_causal_conv1d_fusion.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;

namespace {

bool is_concat_axis_minus_one(const ov::Output<ov::Node>& output) {
    auto concat = ov::as_type_ptr<v0::Concat>(output.get_node_shared_ptr());
    if (!concat) {
        return false;
    }
    const int64_t axis = concat->get_axis();
    if (axis == -1) {
        return true;
    }
    const auto rank = output.get_partial_shape().rank();
    return rank.is_static() && axis == rank.get_length() - 1;
}

bool is_slice_axis_2(const std::shared_ptr<ov::Node>& node) {
    auto slice = ov::as_type_ptr<v8::Slice>(node);
    if (!slice) {
        return false;
    }
    auto axis_const = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(4));
    if (!axis_const) {
        return false;
    }
    auto axis_vec = axis_const->cast_vector<int64_t>();
    return axis_vec.size() == 1 && axis_vec[0] == 2;
}

std::shared_ptr<ov::Node> make_zero_bias(const ov::element::Type& type, size_t channels) {
    std::vector<float> values(channels, 0.0f);
    return v0::Constant::create(type, ov::Shape{channels}, values);
}

}  // namespace

namespace ov::pass {

PagedCausalConv1DFusion::PagedCausalConv1DFusion() {
    MATCHER_SCOPE(PagedCausalConv1DFusion);

    auto token_transpose = wrap_type<v1::Transpose, v1::Reshape>({any_input(), any_input()});
    auto past_state = any_input();
    auto state_concat = wrap_type<v0::Concat>({past_state, token_transpose}, is_concat_axis_minus_one);

    auto weight_const = wrap_type<v0::Constant>();
    auto group_conv = wrap_type<v1::GroupConvolution>({state_concat, weight_const});

    auto slice2 = wrap_type<v8::Slice>({group_conv, any_input(), any_input(), any_input(), any_input()});
    auto swish = wrap_type<v4::Swish>({slice2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pm = m.get_pattern_value_map();

        auto past_state_node = pm.at(past_state);
        auto token_transpose_node = pm.at(token_transpose).get_node_shared_ptr();
        auto state_concat_node = pm.at(state_concat).get_node_shared_ptr();
        auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(group_conv).get_node_shared_ptr());
        auto weight_node = ov::as_type_ptr<v0::Constant>(pm.at(weight_const).get_node_shared_ptr());

        if (!group_conv_node || !weight_node) {
            return false;
        }

        if (!is_slice_axis_2(pm.at(slice2).get_node_shared_ptr())) {
            return false;
        }

        const auto& state_pshape = past_state_node.get_partial_shape();
        if (state_pshape.rank().is_dynamic() || state_pshape.rank().get_length() != 3) {
            return false;
        }

        const auto& weight_pshape = weight_node->get_output_partial_shape(0);
        if (weight_pshape.rank().is_dynamic() || weight_pshape.rank().get_length() != 4 || !weight_pshape.is_static()) {
            return false;
        }

        const auto weight_shape = weight_pshape.get_shape();
        if (weight_shape[1] != 1 || weight_shape[2] != 1) {
            return false;
        }

        if (!state_pshape[1].compatible(weight_shape[0]) || !state_pshape[2].compatible(weight_shape[3])) {
            return false;
        }

        const size_t hidden_size = weight_shape[0];
        const size_t kernel_size = weight_shape[3];

        auto token_node = pm.at(token_transpose);
        const auto& token_pshape = token_node.get_partial_shape();
        if (token_pshape.rank().is_dynamic() || token_pshape.rank().get_length() != 3) {
            return false;
        }

        ov::Output<ov::Node> token_for_reshape = token_node;
        std::shared_ptr<ov::Node> token_transpose_to_2d;
        if (token_pshape[1].compatible(hidden_size)) {
            auto order = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
            token_transpose_to_2d = std::make_shared<v1::Transpose>(token_node, order);
            token_for_reshape = token_transpose_to_2d;
        } else if (!token_pshape[2].compatible(hidden_size)) {
            return false;
        }

        auto input_embeds_shape =
            v0::Constant::create(ov::element::i64,
                                 ov::Shape{2},
                                 std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)});
        auto input_embeds_node = std::make_shared<v1::Reshape>(token_for_reshape, input_embeds_shape, false);

        const auto& strides = group_conv_node->get_strides();
        const auto& pads_begin = group_conv_node->get_pads_begin();
        const auto& pads_end = group_conv_node->get_pads_end();
        const auto& dilations = group_conv_node->get_dilations();
        if (strides != ov::Strides{1} || pads_begin != ov::CoordinateDiff{0} || pads_end != ov::CoordinateDiff{0} ||
            dilations != ov::Strides{1}) {
            return false;
        }

        std::shared_ptr<v8::Slice> state_slice;
        for (const auto& input : state_concat_node->output(0).get_target_inputs()) {
            auto consumer = input.get_node()->shared_from_this();
            auto candidate = ov::as_type_ptr<v8::Slice>(consumer);
            if (!candidate || !is_slice_axis_2(candidate)) {
                continue;
            }
            state_slice = candidate;
            break;
        }
        if (!state_slice) {
            return false;
        }

        auto subsequence_begins = v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
        auto block_indices = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto block_indices_begins = v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
        auto past_lens = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto cache_interval = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});

        auto weight_reshape_shape =
            v0::Constant::create(ov::element::i64,
                                 ov::Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(hidden_size),
                                                      static_cast<int64_t>(1),
                                                      static_cast<int64_t>(kernel_size)});
        auto weight_reshape = std::make_shared<v1::Reshape>(weight_node, weight_reshape_shape, false);

        auto bias_node = make_zero_bias(input_embeds_node->get_output_element_type(0), hidden_size);

        auto paged_conv = std::make_shared<ov::op::internal::PagedCausalConv1D>(
            input_embeds_node,
            past_state_node,
            weight_reshape,
            bias_node,
            subsequence_begins,
            block_indices,
            block_indices_begins,
            past_lens,
            cache_interval);

        paged_conv->set_friendly_name(group_conv_node->get_friendly_name() + "/PagedCausalConv1D");

        auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(paged_conv, unsqueeze_axis);
        unsqueeze->set_friendly_name(group_conv_node->get_friendly_name());

        ov::NodeVector new_nodes;
        new_nodes.reserve(13);
        new_nodes.push_back(input_embeds_shape);
        new_nodes.push_back(input_embeds_node);
        new_nodes.push_back(weight_reshape_shape);
        new_nodes.push_back(weight_reshape);
        new_nodes.push_back(bias_node);
        new_nodes.push_back(subsequence_begins);
        new_nodes.push_back(block_indices);
        new_nodes.push_back(block_indices_begins);
        new_nodes.push_back(past_lens);
        new_nodes.push_back(cache_interval);
        new_nodes.push_back(paged_conv);
        new_nodes.push_back(unsqueeze);
        if (token_transpose_to_2d) {
            new_nodes.push_back(token_transpose_to_2d);
        }
        ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
        ov::replace_node(group_conv_node, unsqueeze);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(swish, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
