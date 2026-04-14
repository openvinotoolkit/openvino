// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"

#include <algorithm>
#include <limits>
#include <map>
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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace {

constexpr const char* CONV_STATE_TABLE_PREFIX = "conv_state_table.";
constexpr const char* MARKED_FOR_PAGED_EXTENSIONS_CLEANUP = "marked_for_paged_extensions_cleanup";
constexpr const char* PAGED_CONV_CACHE_SOURCE = "paged_conv_cache_source";

struct GroupConvFusionResources {
    std::shared_ptr<v0::Parameter> conv_state_table;
    std::string cache_variable_marker;
};

bool is_real_model_conv_group_conv(const std::shared_ptr<v1::GroupConvolution>& group_conv) {
    const auto& name = group_conv->get_friendly_name();
    return name.find(".conv.conv/aten::_convolution/GroupConvolution") != std::string::npos ||
           name.find(".conv/aten::_convolution/GroupConvolution") != std::string::npos;
}

bool is_probable_conv_cache_readvalue(const std::shared_ptr<ov::op::util::ReadValueBase>& rv) {
    const auto& pshape = rv->get_output_partial_shape(0);
    if (!(pshape.rank().is_static() && pshape.rank().get_length() == 3)) {
        return false;
    }
    const auto& rt_info = rv->get_rt_info();
    if (rt_info.count(PAGED_CONV_CACHE_SOURCE) && rt_info.at(PAGED_CONV_CACHE_SOURCE).as<bool>()) {
        return true;
    }
    return rv->get_variable_id().find(".past.conv.") != std::string::npos;
}

void mark_for_paged_extensions_cleanup(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[MARKED_FOR_PAGED_EXTENSIONS_CLEANUP] = true;
}

void mark_assign_sinks_for_cache_marker(const std::shared_ptr<ov::Model>& model,
                                        const std::string& cache_variable_marker) {
    if (cache_variable_marker.empty()) {
        return;
    }
    for (const auto& sink : model->get_sinks()) {
        if (const auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
            if (assign->get_variable_id().find(cache_variable_marker) != std::string::npos) {
                mark_for_paged_extensions_cleanup(sink);
            }
        }
    }
}

std::shared_ptr<ov::Node> find_upstream_cache_param(const ov::Output<ov::Node>& output) {
    std::set<std::shared_ptr<ov::Node>> visited;
    std::vector<std::shared_ptr<ov::Node>> to_visit = {output.get_node_shared_ptr()};

    while (!to_visit.empty()) {
        const auto node = to_visit.back();
        to_visit.pop_back();
        if (!visited.insert(node).second) {
            continue;
        }

        if (const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
            if (is_probable_conv_cache_readvalue(rv)) {
                return node;
            }
        } else if (ov::as_type_ptr<v0::Parameter>(node)) {
            const auto& pshape = node->get_output_partial_shape(0);
            if (pshape.rank().is_static() && pshape.rank().get_length() == 3 && !node->output(0).get_names().empty()) {
                return node;
            }
        }

        // Real models may place cache-state branch on either Concat input, so visit
        // all Concat inputs. For other ops, keep input0 traversal to avoid following
        // auxiliary index paths (e.g. Gather input1).
        if (ov::as_type_ptr<v0::Concat>(node)) {
            for (size_t input_index = 0; input_index < node->get_input_size(); ++input_index) {
                to_visit.push_back(node->get_input_node_shared_ptr(input_index));
            }
        } else if (node->get_input_size() > 0) {
            to_visit.push_back(node->get_input_node_shared_ptr(0));
        }
    }

    return nullptr;
}

std::optional<std::string> extract_cache_variable_marker(const std::shared_ptr<ov::Node>& node) {
    for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
        for (const auto& name : node->output(output_index).get_names()) {
            return name;
        }
    }
    return std::nullopt;
}

struct SharedRuntimeInputs {
    std::shared_ptr<v0::Parameter> subsequence_begins;
    std::shared_ptr<v0::Parameter> block_indices;
    std::shared_ptr<v0::Parameter> block_indices_begins;
    std::shared_ptr<v0::Parameter> past_lens;
    std::shared_ptr<v0::Parameter> cache_interval;
};

struct ParameterCreationResult {
    std::shared_ptr<v0::Parameter> parameter;
    bool created = false;
};

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

ParameterCreationResult create_or_get_named_parameter(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& name,
                                                      const ov::element::Type& type,
                                                      const ov::PartialShape& pshape) {
    for (const auto& input : model->inputs()) {
        if (!input.get_names().count(name)) {
            continue;
        }
        const auto parameter = ov::as_type_ptr<v0::Parameter>(input.get_node_shared_ptr());
        OPENVINO_ASSERT(parameter,
                        "The model is in an inconsistent state. Found input '",
                        name,
                        "', but it is not a v0::Parameter.");
        return {parameter, false};
    }

    const auto parameter = std::make_shared<v0::Parameter>(type, pshape);
    parameter->set_friendly_name(name);
    parameter->get_output_tensor(0).set_names({name});
    model->add_parameters({parameter});
    return {parameter, true};
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
    PagedCausalConv1DFusionMatcher(const SharedRuntimeInputs& shared_inputs,
                                   const std::vector<std::shared_ptr<ov::Node>>& ordered_cache_nodes,
                                   const std::shared_ptr<ov::Model>& model)
        : m_shared_inputs(shared_inputs),
          m_ordered_cache_nodes(ordered_cache_nodes),
          m_model(model) {
        MATCHER_SCOPE(PagedCausalConv1DFusion);

        auto conv_input = any_input();

        auto weight_input = any_input();
        auto group_conv = wrap_type<v1::GroupConvolution>({conv_input, weight_input});

        auto slice2 = wrap_type<v8::Slice>({group_conv, any_input(), any_input(), any_input(), any_input()});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();

            const auto conv_input_output = pm.at(conv_input);
            const auto conv_input_node = conv_input_output.get_node_shared_ptr();
            const auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(group_conv).get_node_shared_ptr());
            const auto weight_node = pm.at(weight_input).get_node_shared_ptr();

            if (!group_conv_node || !weight_node) {
                return false;
            }

            if (!is_slice_axis_2(pm.at(slice2).get_node_shared_ptr())) {
                return false;
            }

            auto fusion_resources_it = m_group_conv_fusion_resources.find(group_conv_node.get());
            if (fusion_resources_it == m_group_conv_fusion_resources.end()) {
                auto cache_param = find_upstream_cache_param(group_conv_node->input_value(0));
                if (!cache_param && is_real_model_conv_group_conv(group_conv_node)) {
                    while (m_fallback_cache_index < m_ordered_cache_nodes.size() &&
                           m_cache_to_state_table.count(m_ordered_cache_nodes[m_fallback_cache_index])) {
                        ++m_fallback_cache_index;
                    }
                    if (m_fallback_cache_index < m_ordered_cache_nodes.size()) {
                        cache_param = m_ordered_cache_nodes[m_fallback_cache_index];
                    }
                }
                if (!cache_param) {
                    return false;
                }

                if (!m_cache_to_state_table.count(cache_param)) {
                    const auto conv_state_table = create_or_get_named_parameter(
                        m_model,
                        make_conv_state_table_name(m_cache_to_state_table.size()),
                        cache_param->get_output_element_type(0),
                        make_conv_state_table_shape(cache_param->get_output_partial_shape(0)));
                    m_cache_to_state_table[cache_param] = conv_state_table.parameter;
                }

                m_group_conv_fusion_resources[group_conv_node.get()] = {
                    m_cache_to_state_table.at(cache_param),
                    extract_cache_variable_marker(cache_param).value_or("")};
                fusion_resources_it = m_group_conv_fusion_resources.find(group_conv_node.get());
            }
            const auto& fusion_resources = fusion_resources_it->second;

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

            if (const auto concat = ov::as_type_ptr<v0::Concat>(conv_input_node)) {
                if (concat->get_input_size() == 2 && is_concat_axis_minus_one(concat->output(0))) {
                    const auto input0 = concat->input_value(0);
                    const auto input1 = concat->input_value(1);
                    const auto input0_cache = find_upstream_cache_param(input0);
                    const auto input1_cache = find_upstream_cache_param(input1);

                    if (input0_cache && !input1_cache) {
                        past_state_output = input0;
                        token_node = input1;
                        state_concat_node = concat;
                    } else if (!input0_cache && input1_cache) {
                        past_state_output = input1;
                        token_node = input0;
                        state_concat_node = concat;
                    }
                }
            }

            const auto& state_pshape = past_state_output.get_node()
                                           ? past_state_output.get_partial_shape()
                                           : fusion_resources.conv_state_table->get_partial_shape();
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
                auto cache_source = find_upstream_cache_param(state_concat_node->output(0));
                const bool stateful_cache = ov::as_type_ptr<ov::op::util::ReadValueBase>(cache_source) != nullptr;
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
                                                                      fusion_resources.conv_state_table,
                                                                      weight_reshape,
                                                                      bias_node,
                                                                      m_shared_inputs.subsequence_begins,
                                                                      m_shared_inputs.block_indices,
                                                                      m_shared_inputs.block_indices_begins,
                                                                      m_shared_inputs.past_lens,
                                                                      m_shared_inputs.cache_interval);

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
            mark_assign_sinks_for_cache_marker(m_model, fusion_resources.cache_variable_marker);
            return true;
        };

        const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(slice2, matcher_name);
        register_matcher(matcher, callback);
    }

private:
    SharedRuntimeInputs m_shared_inputs;
    std::vector<std::shared_ptr<ov::Node>> m_ordered_cache_nodes;
    std::shared_ptr<ov::Model> m_model;
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<v0::Parameter>> m_cache_to_state_table;
    std::map<const ov::Node*, GroupConvFusionResources> m_group_conv_fusion_resources;
    size_t m_fallback_cache_index = 0;
};

}  // namespace

namespace ov::pass {

PagedCausalConv1DFusion::PagedCausalConv1DFusion() {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool PagedCausalConv1DFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PagedCausalConv1DFusion);

    SharedRuntimeInputs shared_inputs{
        create_or_get_named_parameter(model, "subsequence_begins", ov::element::i32, ov::PartialShape{-1}).parameter,
        create_or_get_named_parameter(model, "la.block_indices", ov::element::i32, ov::PartialShape{-1})
            .parameter,
        create_or_get_named_parameter(model, "la.block_indices_begins", ov::element::i32, ov::PartialShape{-1})
            .parameter,
        create_or_get_named_parameter(model, "la.past_lens", ov::element::i32, ov::PartialShape{-1}).parameter,
        create_or_get_named_parameter(model, "la.cache_interval", ov::element::i32, ov::PartialShape{-1})
            .parameter};

    std::vector<std::shared_ptr<ov::Node>> ordered_cache_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
            if (is_probable_conv_cache_readvalue(rv)) {
                ordered_cache_nodes.push_back(rv);
            }
            continue;
        }
        const auto param = ov::as_type_ptr<v0::Parameter>(node);
        if (!param) {
            continue;
        }
        const auto& pshape = param->get_output_partial_shape(0);
        if (!(pshape.rank().is_static() && pshape.rank().get_length() == 3)) {
            continue;
        }
        if (!param->output(0).get_names().empty() && param->output(0).get_target_inputs().empty()) {
            ordered_cache_nodes.push_back(param);
        }
    }

    ov::pass::Manager manager(get_pass_config(), "PagedCausalConv1DFusion");
    manager.set_per_pass_validation(false);
    manager.register_pass<PagedCausalConv1DFusionMatcher>(shared_inputs, ordered_cache_nodes, model);
    const bool rewritten = manager.run_passes(model);

    return rewritten;
}

}  // namespace ov::pass
