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
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace {

constexpr const char* CONV_STATE_TABLE_PREFIX = "conv_state_table.";

bool is_conv_cache_readvalue(const std::shared_ptr<ov::op::util::ReadValueBase>& rv) {
    const auto& pshape = rv->get_output_partial_shape(0);
    return pshape.rank().is_static() && pshape.rank().get_length() == 3;
}

bool is_concat_axis_minus_one(const ov::Output<ov::Node>& output) {
    const auto concat = ov::as_type_ptr<v0::Concat>(output.get_node_shared_ptr());
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
    const auto slice = ov::as_type_ptr<v8::Slice>(node);
    if (!slice) {
        return false;
    }
    const auto axis_const = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(4));
    if (!axis_const) {
        return false;
    }
    const auto axis_vec = axis_const->cast_vector<int64_t>();
    return axis_vec.size() == 1 && (axis_vec[0] == 2 || axis_vec[0] == -1);
}

std::shared_ptr<ov::Node> make_zero_bias(const ov::element::Type& type, const size_t channels) {
    std::vector<float> values(channels, 0.0f);
    return v0::Constant::create(type, ov::Shape{channels}, values);
}

std::string make_conv_state_table_name(const size_t layer_index) {
    return std::string(CONV_STATE_TABLE_PREFIX) + std::to_string(layer_index);
}

ov::PartialShape make_conv_state_table_shape(const ov::PartialShape& past_state_shape) {
    if (past_state_shape.rank().is_static() && past_state_shape.rank().get_length() == 3) {
        return ov::PartialShape{ov::Dimension::dynamic(), past_state_shape[1], past_state_shape[2]};
    }
    return ov::PartialShape::dynamic(3);
}

class PagedCausalConv1DFusionMatcher : public ov::pass::MatcherPass {
public:
    PagedCausalConv1DFusionMatcher(ov::pass::paged_attention::PaParams& pa_params,
                                   std::unordered_set<std::string>& var_ids_to_remove)
        : m_params(pa_params),
          m_var_ids_to_remove(var_ids_to_remove) {
        MATCHER_SCOPE(PagedCausalConv1DFusion);

        // ReadValue (rank-3) → optional Gather → Concat(past, token) OR Concat(token, past)
        auto read_value = wrap_type<ov::op::util::ReadValueBase>([](const ov::Output<ov::Node>& out) {
            const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(out.get_node_shared_ptr());
            return rv && is_conv_cache_readvalue(rv);
        });
        auto past_via_gather = ov::pass::pattern::optional<v8::Gather>({read_value, any_input(), any_input()});
        auto token_input = any_input();
        auto token_input_rev = any_input();
        auto state_concat_past_first = wrap_type<v0::Concat>({past_via_gather, token_input});
        auto state_concat_token_first = wrap_type<v0::Concat>({token_input_rev, past_via_gather});
        auto state_concat = std::make_shared<ov::pass::pattern::op::Or>(
            ov::OutputVector{state_concat_past_first->output(0), state_concat_token_first->output(0)});

        auto weight_input = any_input();
        auto group_conv = wrap_type<v1::GroupConvolution>({state_concat, weight_input});

        auto slice2 = wrap_type<v8::Slice>({group_conv, any_input(), any_input(), any_input(), any_input()});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();

            const auto conv_input_output = pm.at(state_concat);
            const auto conv_input_node = conv_input_output.get_node_shared_ptr();
            const auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(group_conv).get_node_shared_ptr());
            const auto weight_node = pm.at(weight_input).get_node_shared_ptr();

            if (!group_conv_node || !weight_node) {
                return false;
            }

            if (!is_slice_axis_2(pm.at(slice2).get_node_shared_ptr())) {
                return false;
            }

            if (!pm.count(read_value)) {
                return false;
            }
            const auto cache_param = pm.at(read_value).get_node_shared_ptr();
            const auto cache_rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(cache_param);
            OPENVINO_ASSERT(cache_rv, "Matched cache node is expected to be ReadValue");

            const size_t layer_index = static_cast<size_t>(m_layer_index++);
            const auto conv_state_table =
                m_params.add(make_conv_state_table_name(layer_index),
                             cache_param->get_output_element_type(0),
                             make_conv_state_table_shape(cache_param->get_output_partial_shape(0)));
            enable_keep_const_precision(conv_state_table);
            m_var_ids_to_remove.insert(cache_rv->get_variable_id());

            const auto& weight_pshape = weight_node->get_output_partial_shape(0);
            if (weight_pshape.rank().is_dynamic() || weight_pshape.rank().get_length() != 4 ||
                !weight_pshape.is_static()) {
                return false;
            }

            const auto weight_shape = weight_pshape.get_shape();
            if (weight_shape[1] != 1 || weight_shape[2] != 1) {
                return false;
            }

            ov::Output<ov::Node> token_node = conv_input_output;
            ov::Output<ov::Node> past_state_output;
            std::shared_ptr<v0::Concat> state_concat_node;

            if (pm.count(state_concat_past_first)) {
                state_concat_node = ov::as_type_ptr<v0::Concat>(pm.at(state_concat_past_first).get_node_shared_ptr());
                OPENVINO_ASSERT(state_concat_node, "state_concat_past_first is expected to be Concat");
                past_state_output = state_concat_node->input_value(0);
                token_node = state_concat_node->input_value(1);
            } else if (pm.count(state_concat_token_first)) {
                state_concat_node = ov::as_type_ptr<v0::Concat>(pm.at(state_concat_token_first).get_node_shared_ptr());
                OPENVINO_ASSERT(state_concat_node, "state_concat_token_first is expected to be Concat");
                past_state_output = state_concat_node->input_value(1);
                token_node = state_concat_node->input_value(0);
            } else if (const auto concat = ov::as_type_ptr<v0::Concat>(conv_input_node)) {
                if (concat->get_input_size() == 2 && is_concat_axis_minus_one(concat->output(0))) {
                    state_concat_node = concat;
                }
            }

            const auto& state_pshape = past_state_output.get_node() ? past_state_output.get_partial_shape()
                                                                    : conv_state_table->get_partial_shape();
            if (state_pshape.rank().is_dynamic() || state_pshape.rank().get_length() != 3) {
                return false;
            }

            if (!state_pshape[1].compatible(weight_shape[0]) || !state_pshape[2].compatible(weight_shape[3])) {
                return false;
            }

            const size_t hidden_size = weight_shape[0];
            const size_t kernel_size = weight_shape[3];

            const auto& token_pshape = token_node.get_partial_shape();
            if (token_pshape.rank().is_dynamic() || token_pshape.rank().get_length() != 3) {
                return false;
            }

            ov::Output<ov::Node> token_for_reshape = token_node;
            std::shared_ptr<ov::Node> token_transpose_to_2d;
            if (token_pshape[1].compatible(hidden_size)) {
                const auto order = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
                token_transpose_to_2d = std::make_shared<v1::Transpose>(token_node, order);
                token_for_reshape = token_transpose_to_2d;
            } else if (!token_pshape[2].compatible(hidden_size)) {
                return false;
            }

            const auto input_embeds_shape =
                v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)});
            const auto input_embeds_node = std::make_shared<v1::Reshape>(token_for_reshape, input_embeds_shape, false);

            const auto& strides = group_conv_node->get_strides();
            const auto& pads_begin = group_conv_node->get_pads_begin();
            const auto& pads_end = group_conv_node->get_pads_end();
            const auto& dilations = group_conv_node->get_dilations();
            if (strides != ov::Strides{1} || dilations != ov::Strides{1}) {
                return false;
            }
            if (pads_begin.size() != 1 || pads_end.size() != 1 || pads_begin[0] != pads_end[0]) {
                return false;
            }

            std::shared_ptr<v8::Slice> state_slice;
            if (state_concat_node) {
                for (const auto& input : state_concat_node->output(0).get_target_inputs()) {
                    const auto consumer = input.get_node()->shared_from_this();
                    const auto candidate = ov::as_type_ptr<v8::Slice>(consumer);
                    if (!candidate || !is_slice_axis_2(candidate)) {
                        continue;
                    }
                    state_slice = candidate;
                    break;
                }

                // Keep strict requirement for non-stateful cache sources (Parameter-based paths),
                // but allow stateful ReadValue-based real-model paths without legacy state slice.
                const bool stateful_cache = pm.count(read_value) > 0;
                if (!state_slice && !stateful_cache) {
                    return false;
                }
            }

            const auto weight_reshape_shape =
                v0::Constant::create(ov::element::i64,
                                     ov::Shape{3},
                                     std::vector<int64_t>{static_cast<int64_t>(hidden_size),
                                                          static_cast<int64_t>(1),
                                                          static_cast<int64_t>(kernel_size)});
            const auto weight_reshape = std::make_shared<v1::Reshape>(weight_node, weight_reshape_shape, false);

            const auto bias_node = make_zero_bias(input_embeds_node->get_output_element_type(0), hidden_size);

            const auto paged_conv =
                std::make_shared<ov::op::internal::PagedCausalConv1D>(input_embeds_node,
                                                                      conv_state_table,
                                                                      weight_reshape,
                                                                      bias_node,
                                                                      m_params["subsequence_begins"],
                                                                      m_params["la.block_indices"],
                                                                      m_params["la.block_indices_begins"],
                                                                      m_params["la.past_lens"],
                                                                      m_params["la.cache_interval"]);

            paged_conv->set_friendly_name(group_conv_node->get_friendly_name() + "/PagedCausalConv1D");

            const auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
            const auto unsqueeze = std::make_shared<v0::Unsqueeze>(paged_conv, unsqueeze_axis);
            unsqueeze->set_friendly_name(group_conv_node->get_friendly_name());

            // Keep legacy concat branch independent from new conv_state_table parameter.
            // Old present-state consumers are rewired to the original past-state data path,
            // while conv_state_table remains connected only to PagedCausalConv1D input[1].
            if (state_slice && past_state_output.get_node()) {
                state_slice->output(0).replace(past_state_output);
            }

            ov::NodeVector new_nodes;
            new_nodes.reserve(8);
            new_nodes.push_back(input_embeds_shape);
            new_nodes.push_back(input_embeds_node);
            new_nodes.push_back(weight_reshape_shape);
            new_nodes.push_back(weight_reshape);
            new_nodes.push_back(bias_node);
            new_nodes.push_back(paged_conv);
            new_nodes.push_back(unsqueeze);
            if (token_transpose_to_2d) {
                new_nodes.push_back(token_transpose_to_2d);
            }
            ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
            ov::replace_node(pm.at(slice2).get_node_shared_ptr(), unsqueeze);
            return true;
        };

        const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(slice2, matcher_name);
        register_matcher(matcher, callback);
    }

private:
    ov::pass::paged_attention::PaParams& m_params;
    std::unordered_set<std::string>& m_var_ids_to_remove;
    int m_layer_index = 0;
};

}  // namespace

namespace ov::pass {

PagedCausalConv1DFusion::PagedCausalConv1DFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                 const ov::pass::paged_attention::Options& options,
                                                 std::unordered_set<std::string>& var_ids_to_remove)
    : m_params(pa_params),
      m_options(options),
      m_var_ids_to_remove(var_ids_to_remove) {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool PagedCausalConv1DFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PagedCausalConv1DFusion);

    m_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

    ov::pass::Manager manager(get_pass_config(), "PagedCausalConv1DFusion");
    manager.set_per_pass_validation(false);
    manager.register_pass<PagedCausalConv1DFusionMatcher>(m_params, m_var_ids_to_remove);
    const bool rewritten = manager.run_passes(model);

    if (rewritten) {
        model->add_parameters(m_params.items());
    }

    return rewritten;
}

}  // namespace ov::pass
