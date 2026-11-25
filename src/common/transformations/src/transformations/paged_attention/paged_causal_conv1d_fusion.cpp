// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::has_static_rank;
using ov::pass::pattern::has_static_shape;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;

namespace {

// PagedCausalConv1D has no stride/padding attributes: it always produces one output per
// input token. A strided or padded GroupConvolution would change the output token count
// and is not representable by the fused op.
ov::pass::pattern::op::Predicate has_unit_stride_and_no_padding() {
    return ov::pass::pattern::op::Predicate(
        [](const ov::Output<ov::Node>& output) -> bool {
            const auto group_conv = ov::as_type_ptr<v1::GroupConvolution>(output.get_node_shared_ptr());
            const auto& strides = group_conv->get_strides();
            const auto& pads_begin = group_conv->get_pads_begin();
            const auto& pads_end = group_conv->get_pads_end();
            return (strides.empty() || strides[0] == 1) && (pads_begin.empty() || pads_begin[0] == 0) &&
                   (pads_end.empty() || pads_end[0] == 0);
        },
        "has_unit_stride_and_no_padding");
}

}  // namespace

namespace ov::pass {

class PagedCausalConv1DFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedCausalConv1DFusionMatcher");
    PagedCausalConv1DFusionMatcher(ov::pass::paged_attention::PaParams& pa_params,
                                   std::unordered_set<std::string>& var_ids_to_remove,
                                   size_t& layer_index);
};

PagedCausalConv1DFusionMatcher::PagedCausalConv1DFusionMatcher(ov::pass::paged_attention::PaParams& pa_params,
                                                               std::unordered_set<std::string>& var_ids_to_remove,
                                                               size_t& layer_index) {
    MATCHER_SCOPE(PagedCausalConv1DFusionMatcher);

    auto p_read_value = wrap_type<ov::op::util::ReadValueBase>(has_static_rank() && rank_equals(3));
    auto p_past_via_gather =
        ov::pass::pattern::optional<ov::op::util::GatherBase>({p_read_value, any_input(), any_input()});
    auto p_token_input = any_input(rank_equals(3));
    auto p_concat_past_first = wrap_type<v0::Concat>({p_past_via_gather, p_token_input}, {{"axis", -1}});
    auto p_concat_token_first = wrap_type<v0::Concat>({p_token_input, p_past_via_gather}, {{"axis", -1}});
    auto p_state_concat = p_concat_past_first | p_concat_token_first;

    auto p_weight_input = any_input(has_static_shape() && rank_equals(4));
    auto p_group_conv =
        wrap_type<v1::GroupConvolution>({p_state_concat, p_weight_input}, has_unit_stride_and_no_padding());

    auto p_bias_input = any_input(has_static_shape());
    auto p_add_bias = ov::pass::pattern::optional<v1::Add>({p_group_conv, p_bias_input});
    auto p_slice_out =
        ov::pass::pattern::optional<v8::Slice>({p_add_bias, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove, &layer_index](
                                             ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pm = m.get_pattern_value_map();
        const bool with_output_slice = pm.count(p_slice_out) != 0;

        const auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(p_group_conv).get_node_shared_ptr());
        const auto weight_node = pm.at(p_weight_input).get_node_shared_ptr();
        const auto cache_rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(p_read_value).get_node_shared_ptr());
        const auto past_state = pm.count(p_past_via_gather) ? pm.at(p_past_via_gather).get_node_shared_ptr() : cache_rv;
        auto token_input = pm.at(p_token_input).get_node_shared_ptr();

        OPENVINO_ASSERT(group_conv_node);
        OPENVINO_ASSERT(cache_rv);

        const auto& weight_shape = weight_node->get_output_shape(0);
        const size_t hidden_size = weight_shape[0];
        const size_t kernel_size = weight_shape[3];

        const size_t dilation = group_conv_node->get_dilations().size() > 0 ? group_conv_node->get_dilations()[0] : 1;
        const size_t causal_window = (kernel_size - 1) * dilation + 1;
        // A full-window history produces one extra output, removed by the output Slice.
        // The history-only form (e.g. PLE) already produces one output per current token.
        const size_t history_length = with_output_slice ? causal_window : causal_window - 1;

        const auto& state_pshape = past_state->get_output_partial_shape(0);
        // Without output trimming, history must precede the tokens and have an exact length.
        // TODO: remove the case where first goes present then past (it pobably doesn't exist in real life)
        if (!with_output_slice && (!pm.count(p_concat_past_first) || state_pshape[2].is_dynamic())) {
            return false;
        }
        if (!state_pshape[1].compatible(hidden_size) || !state_pshape[2].compatible(history_length)) {
            return false;
        }

        const auto& token_pshape = token_input->get_output_partial_shape(0);
        if (token_pshape[1].compatible(hidden_size)) {
            const auto order = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
            token_input = std::make_shared<v1::Transpose>(token_input, order);
        } else if (!token_pshape[2].compatible(hidden_size)) {
            return false;
        }

        const auto input_embeds_shape =
            v0::Constant::create(ov::element::i64,
                                 ov::Shape{2},
                                 std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)});
        const auto input_embeds_node = std::make_shared<v1::Reshape>(token_input, input_embeds_shape, false);

        const auto pa_weight_shape = v0::Constant::create(ov::element::i64,
                                                          ov::Shape{3},
                                                          std::vector<int64_t>{static_cast<int64_t>(hidden_size),
                                                                               static_cast<int64_t>(1),
                                                                               static_cast<int64_t>(kernel_size)});
        const auto weight_reshaped = std::make_shared<v1::Reshape>(weight_node, pa_weight_shape, false);

        const auto& elem_type = input_embeds_node->get_output_element_type(0);
        std::shared_ptr<ov::Node> bias_node;
        if (pm.count(p_add_bias)) {
            const auto bias_input = pm.at(p_bias_input).get_node_shared_ptr();
            const auto& bias_shape = bias_input->get_output_partial_shape(0);
            if (bias_shape.rank().get_length() == 1 && bias_shape[0].compatible(hidden_size)) {
                bias_node = bias_input;
            } else {
                if (bias_shape.is_static() && ov::shape_size(bias_shape.to_shape()) != hidden_size) {
                    // Validate that the total element count matches hidden_size
                    return false;
                }
                const auto bias_shape_node =
                    v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
                bias_node = std::make_shared<v1::Reshape>(bias_input, bias_shape_node, false);
            }
        } else {
            bias_node = v0::Constant::create(elem_type, ov::Shape{0}, std::vector<float>{});
        }

        pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

        // The paged kernel shifts before inserting the current token and always stores
        // the full window, even when the source graph only stores window - 1 inputs.
        const auto conv_state_table = pa_params.add("conv_state_table." + std::to_string(layer_index++),
                                                    ov::element::dynamic,
                                                    ov::PartialShape{ov::Dimension::dynamic(),
                                                                     static_cast<int64_t>(hidden_size),
                                                                     static_cast<int64_t>(causal_window)});

        enable_keep_const_precision(conv_state_table);
        var_ids_to_remove.insert(cache_rv->get_variable_id());

        const auto paged_conv =
            std::make_shared<ov::op::internal::PagedCausalConv1D>(input_embeds_node,
                                                                  conv_state_table,
                                                                  weight_reshaped,
                                                                  bias_node,
                                                                  pa_params["subsequence_begins"],
                                                                  pa_params["la.block_indices"],
                                                                  pa_params["la.block_indices_begins"],
                                                                  pa_params["la.past_lens"],
                                                                  pa_params["la.cache_interval"],
                                                                  dilation);

        paged_conv->set_friendly_name(group_conv_node->get_friendly_name() + "/PagedCausalConv1D");

        const auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto unsqueeze = std::make_shared<v0::Unsqueeze>(paged_conv, unsqueeze_axis);
        unsqueeze->set_friendly_name(group_conv_node->get_friendly_name());

        ov::NodeVector new_nodes{input_embeds_shape,
                                 input_embeds_node,
                                 pa_weight_shape,
                                 weight_reshaped,
                                 bias_node,
                                 paged_conv,
                                 unsqueeze};

        ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
        ov::replace_node(m.get_match_root(), unsqueeze);
        return true;
    };

    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(p_slice_out, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass

ov::pass::PagedCausalConv1DFusion::PagedCausalConv1DFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                           std::unordered_set<std::string>& var_ids_to_remove) {
    add_matcher<PagedCausalConv1DFusionMatcher>(pa_params, var_ids_to_remove, m_layer_index);
}

ov::pass::PagedCausalConv1DWithDynamicTaps::PagedCausalConv1DWithDynamicTaps(
    ov::pass::paged_attention::PaParams& pa_params,
    std::unordered_set<std::string>& var_ids_to_remove)
    : m_pa_params(pa_params),
      m_var_ids_to_remove(var_ids_to_remove) {}

bool ov::pass::PagedCausalConv1DWithDynamicTaps::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PagedCausalConv1DWithDynamicTaps);

    const auto constant_is = [](const ov::Output<ov::Node>& value, const std::vector<int64_t>& expected) {
        const auto constant = ov::as_type_ptr<v0::Constant>(value.get_node_shared_ptr());
        return constant && constant->get_element_type().is_integral_number() &&
               constant->cast_vector<int64_t>() == expected;
    };
    const auto i64 = [](const std::vector<int64_t>& values) {
        return v0::Constant::create(ov::element::i64, {values.size()}, values);
    };
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto window = ov::as_type_ptr<v0::Concat>(node);
        if (!window || window->get_axis() != 3 || window->get_input_size() != 2)
            continue;
        const auto state = ov::as_type_ptr<ov::op::util::ReadValueBase>(window->get_input_node_shared_ptr(0));
        const auto token_transpose = ov::as_type_ptr<v1::Transpose>(window->get_input_node_shared_ptr(1));
        if (!state || !token_transpose || state->output(0).get_target_inputs().size() != 1 ||
            !constant_is(token_transpose->input_value(1), {0, 1, 3, 2}))
            continue;
        const auto state_shape = state->get_output_partial_shape(0);
        const auto token_shape = token_transpose->get_input_partial_shape(0);
        if (!state_shape.is_static() || state_shape.rank() != 4 || state_shape[0] != 1 || state_shape[1] != 1 ||
            state_shape[2].get_length() <= 0 || state_shape[3].get_length() <= 0 || token_shape.rank() != 4 ||
            token_shape[1] != 1 || token_shape[2] != 1 || token_shape[3] != state_shape[2])
            continue;
        const auto channels = state_shape[2].get_length();
        const auto history = state_shape[3].get_length();
        OPENVINO_ASSERT(history <= std::numeric_limits<int64_t>::max() - 3,
                        "Paged causal convolution history is too large");
        const auto kernel = history + 1;
        if (window->output(0).get_target_inputs().size() != static_cast<size_t>(kernel + 2))
            continue;

        std::shared_ptr<v1::GroupConvolution> convolution;
        std::shared_ptr<v8::Slice> update;
        std::vector<std::shared_ptr<v8::Slice>> taps(static_cast<size_t>(kernel));
        bool matches = true;
        for (const auto& target : window->output(0).get_target_inputs()) {
            if (const auto reshape = ov::as_type_ptr<v1::Reshape>(target.get_node()->shared_from_this())) {
                if (target.get_index() != 0 || reshape->output(0).get_target_inputs().size() != 1 || convolution) {
                    matches = false;
                    break;
                }
                const auto conv_target = *reshape->output(0).get_target_inputs().begin();
                convolution = ov::as_type_ptr<v1::GroupConvolution>(conv_target.get_node()->shared_from_this());
                if (!constant_is(reshape->input_value(1), {1, channels, -1}) || reshape->get_special_zero() ||
                    !convolution || conv_target.get_index() != 0 ||
                    convolution->get_input_partial_shape(1) != ov::PartialShape({channels, 1, 1, kernel}) ||
                    convolution->get_strides() != ov::Strides{1} || convolution->get_dilations() != ov::Strides{1} ||
                    convolution->get_pads_begin() != ov::CoordinateDiff{0} ||
                    convolution->get_pads_end() != ov::CoordinateDiff{0} ||
                    convolution->get_auto_pad() != ov::op::PadType::EXPLICIT) {
                    matches = false;
                    break;
                }
                continue;
            }
            const auto slice = ov::as_type_ptr<v8::Slice>(target.get_node()->shared_from_this());
            if (!slice || target.get_index() != 0 || slice->get_input_size() != 5 ||
                !constant_is(slice->input_value(3), {1}) || !constant_is(slice->input_value(4), {3})) {
                matches = false;
                break;
            }
            if (constant_is(slice->input_value(1), {1 - kernel}) &&
                constant_is(slice->input_value(2), {std::numeric_limits<int64_t>::max()})) {
                const auto consumers = slice->output(0).get_target_inputs();
                if (update || consumers.size() != 1) {
                    matches = false;
                    break;
                }
                const auto assign =
                    ov::as_type_ptr<ov::op::util::AssignBase>(consumers.begin()->get_node()->shared_from_this());
                if (!assign || assign->get_variable_id() != state->get_variable_id()) {
                    matches = false;
                    break;
                }
                update = slice;
                continue;
            }

            const auto begin = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(1));
            const auto end = ov::as_type_ptr<v1::Add>(slice->get_input_node_shared_ptr(2));
            if (!begin || !begin->get_element_type().is_integral_number() || ov::shape_size(begin->get_shape()) != 1 ||
                !end || end->input_value(0) != slice->input_value(1)) {
                matches = false;
                break;
            }
            const auto start = begin->cast_vector<int64_t>()[0];
            const auto length = ov::as_type_ptr<v8::Gather>(end->get_input_node_shared_ptr(1));
            const auto shape = length ? ov::as_type_ptr<v3::ShapeOf>(length->get_input_node_shared_ptr(0)) : nullptr;
            if (start < 0 || start >= kernel || !length || !shape ||
                shape->input_value(0) != token_transpose->input_value(0) || !constant_is(length->input_value(1), {2}) ||
                !constant_is(length->input_value(2), {0}) || taps[static_cast<size_t>(kernel - 1 - start)]) {
                matches = false;
                break;
            }
            taps[static_cast<size_t>(kernel - 1 - start)] = slice;
        }
        if (!matches || !convolution || !update || std::any_of(taps.begin(), taps.end(), [](const auto& tap) {
                return !tap;
            }))
            continue;

        const auto channel_count = static_cast<size_t>(channels);
        const auto kernel_size = static_cast<size_t>(kernel);
        OPENVINO_ASSERT(channels <= std::numeric_limits<int64_t>::max() / (kernel + 1) &&
                            channel_count <= std::numeric_limits<size_t>::max() / kernel_size / kernel_size,
                        "Paged causal convolution tap weights are too large");
        std::vector<float> tap_weights(channel_count * kernel_size * kernel_size, 0.f);
        for (size_t lag = 0; lag < kernel_size; ++lag) {
            for (size_t channel = 0; channel < channel_count; ++channel)
                tap_weights[(lag * channel_count + channel) * kernel_size + kernel_size - 1 - lag] = 1.f;
        }
        const auto rows = std::make_shared<v1::Reshape>(token_transpose->input_value(0), i64({-1, channels}), false);
        ov::OutputVector repeated_inputs(kernel_size + 1, rows);
        const auto expanded = std::make_shared<v0::Concat>(repeated_inputs, 1);
        const auto static_weights =
            std::make_shared<v1::Reshape>(convolution->input_value(1), i64({channels, 1, kernel}), false);
        const auto one_hot = v0::Constant::create(convolution->get_input_element_type(1),
                                                  {kernel_size * channel_count, 1, kernel_size},
                                                  tap_weights);
        const auto weights = std::make_shared<v0::Concat>(ov::OutputVector{static_weights, one_hot}, 0);
        const auto bias = v0::Constant::create(rows->get_element_type(), {0}, std::vector<float>{});
        m_pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
        m_pa_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
        m_pa_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
        m_pa_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
        m_pa_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});
        size_t index = 0;
        while (m_pa_params.exists("conv_state_table." + std::to_string(index)))
            ++index;
        const auto table = m_pa_params.add("conv_state_table." + std::to_string(index),
                                           ov::element::dynamic,
                                           ov::PartialShape{-1, channels * (kernel + 1), kernel});
        enable_keep_const_precision(table);
        const auto paged = std::make_shared<ov::op::internal::PagedCausalConv1D>(expanded,
                                                                                 table,
                                                                                 weights,
                                                                                 bias,
                                                                                 m_pa_params["subsequence_begins"],
                                                                                 m_pa_params["la.block_indices"],
                                                                                 m_pa_params["la.block_indices_begins"],
                                                                                 m_pa_params["la.past_lens"],
                                                                                 m_pa_params["la.cache_interval"],
                                                                                 1);
        paged->set_friendly_name(convolution->get_friendly_name() + "/PagedCausalConv1D");

        const auto column_slice = [&](int64_t start) {
            return std::make_shared<v8::Slice>(paged, i64({start}), i64({start + channels}), i64({1}), i64({1}));
        };
        const auto static_slice = column_slice(0);
        const auto static_output = std::make_shared<v0::Unsqueeze>(static_slice, i64({2}));
        static_output->set_friendly_name(convolution->get_friendly_name());
        ov::NodeVector
            new_nodes{rows, expanded, static_weights, one_hot, weights, bias, paged, static_slice, static_output};
        for (size_t lag = 0; lag < kernel_size; ++lag) {
            const auto lag_slice = column_slice(static_cast<int64_t>(lag + 1) * channels);
            const auto lag_output = std::make_shared<v1::Reshape>(lag_slice, i64({0, 1, channels, 1}), true);
            lag_output->set_friendly_name(taps[lag]->get_friendly_name());
            new_nodes.push_back(lag_slice);
            new_nodes.push_back(lag_output);
            ov::replace_node(taps[lag], lag_output);
        }
        ov::copy_runtime_info({state, window, convolution, update}, new_nodes);
        ov::replace_node(convolution, static_output);
        m_var_ids_to_remove.insert(state->get_variable_id());
        changed = true;
    }
    return changed;
}
