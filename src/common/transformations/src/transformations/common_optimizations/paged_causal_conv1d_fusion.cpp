// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/paged_causal_conv1d_fusion.hpp"

#include <algorithm>
#include <cctype>
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
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
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

constexpr const char* CACHE_PARAMS_PAST_CONV_PREFIX = "cache_params.past.conv.";
constexpr const char* CONV_STATE_TABLE_PREFIX = "conv_state_table.";

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

std::optional<size_t> parse_layer_index_from_name(const std::string& name, const std::string& prefix) {
    if (name.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }
    const std::string suffix = name.substr(prefix.size());
    if (suffix.empty() ||
        !std::all_of(suffix.begin(), suffix.end(), [](const char value) { return std::isdigit(value) != 0; })) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::stoull(suffix));
}

std::optional<size_t> extract_conv_layer_index(const ov::Output<ov::Node>& output) {
    for (const auto& name : output.get_names()) {
        if (const auto layer_index = parse_layer_index_from_name(name, CACHE_PARAMS_PAST_CONV_PREFIX)) {
            return layer_index;
        }
    }
    return parse_layer_index_from_name(output.get_node()->get_friendly_name(), CACHE_PARAMS_PAST_CONV_PREFIX);
}

std::optional<size_t> extract_conv_layer_index_upstream(const ov::Output<ov::Node>& output) {
    if (const auto direct = extract_conv_layer_index(output)) {
        return direct;
    }

    std::set<std::shared_ptr<ov::Node>> visited;
    std::vector<std::shared_ptr<ov::Node>> to_visit;
    to_visit.push_back(output.get_node_shared_ptr());

    while (!to_visit.empty()) {
        const auto node = to_visit.back();
        to_visit.pop_back();
        if (!visited.insert(node).second) {
            continue;
        }

        for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
            if (const auto layer_index = extract_conv_layer_index(node->output(output_index))) {
                return layer_index;
            }
        }

        // Follow only data-flow input 0 to avoid any dependency on auxiliary index tensors
        // (e.g. beam_idx on Gather input 1).
        if (node->get_input_size() > 0) {
            to_visit.push_back(node->get_input_node_shared_ptr(0));
        }
    }

    return std::nullopt;
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
                                  const std::map<size_t, std::shared_ptr<v0::Parameter>>& conv_state_tables) {
        MATCHER_SCOPE(PagedCausalConv1DFusion);

        auto token_input = any_input();
        auto past_state = any_input();
        auto state_concat = wrap_type<v0::Concat>({past_state, token_input}, is_concat_axis_minus_one);

        auto weight_const = wrap_type<v0::Constant>();
        auto group_conv = wrap_type<v1::GroupConvolution>({state_concat, weight_const});

        auto slice2 = wrap_type<v8::Slice>({group_conv, any_input(), any_input(), any_input(), any_input()});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();

            const auto past_state_output = pm.at(past_state);
            const auto token_input_node = pm.at(token_input).get_node_shared_ptr();
            const auto state_concat_node = pm.at(state_concat).get_node_shared_ptr();
            const auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(group_conv).get_node_shared_ptr());
            const auto weight_node = ov::as_type_ptr<v0::Constant>(pm.at(weight_const).get_node_shared_ptr());

            if (!group_conv_node || !weight_node) {
                return false;
            }

            if (!is_slice_axis_2(pm.at(slice2).get_node_shared_ptr())) {
                return false;
            }

            const auto layer_index = extract_conv_layer_index_upstream(past_state_output);
            if (!layer_index) {
                return false;
            }

            const auto conv_state_table_it = conv_state_tables.find(*layer_index);
            if (conv_state_table_it == conv_state_tables.end()) {
                return false;
            }

            const auto& state_pshape = past_state_output.get_partial_shape();
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

            ov::Output<ov::Node> token_node = token_input_node;
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
            if (strides != ov::Strides{1} || pads_begin != ov::CoordinateDiff{0} || pads_end != ov::CoordinateDiff{0} ||
                dilations != ov::Strides{1}) {
                return false;
            }

            std::shared_ptr<v8::Slice> state_slice;
            for (const auto& input : state_concat_node->output(0).get_target_inputs()) {
                const auto consumer = input.get_node()->shared_from_this();
                const auto candidate = ov::as_type_ptr<v8::Slice>(consumer);
                if (!candidate || !is_slice_axis_2(candidate)) {
                    continue;
                }
                state_slice = candidate;
                break;
            }
            if (!state_slice) {
                return false;
            }

            const auto weight_reshape_shape =
                v0::Constant::create(ov::element::i64,
                                     ov::Shape{3},
                                     std::vector<int64_t>{static_cast<int64_t>(hidden_size),
                                                          static_cast<int64_t>(1),
                                                          static_cast<int64_t>(kernel_size)});
            const auto weight_reshape = std::make_shared<v1::Reshape>(weight_node, weight_reshape_shape, false);

            const auto bias_node = make_zero_bias(input_embeds_node->get_output_element_type(0), hidden_size);

            const auto paged_conv = std::make_shared<ov::op::internal::PagedCausalConv1D>(input_embeds_node,
                                                                                           conv_state_table_it->second,
                                                                                           weight_reshape,
                                                                                           bias_node,
                                                                                           shared_inputs.subsequence_begins,
                                                                                           shared_inputs.block_indices,
                                                                                           shared_inputs.block_indices_begins,
                                                                                           shared_inputs.past_lens,
                                                                                           shared_inputs.cache_interval);

            paged_conv->set_friendly_name(group_conv_node->get_friendly_name() + "/PagedCausalConv1D");

            const auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
            const auto unsqueeze = std::make_shared<v0::Unsqueeze>(paged_conv, unsqueeze_axis);
            unsqueeze->set_friendly_name(group_conv_node->get_friendly_name());

            // Keep legacy branch independent from new conv_state_table parameter.
            // Old present-state consumers are rewired to the original past-state data path,
            // while conv_state_table remains connected only to PagedCausalConv1D input[1].
            state_slice->output(0).replace(past_state_output);

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
            ov::replace_node(group_conv_node, unsqueeze);
            return true;
        };

        const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(slice2, matcher_name);
        register_matcher(matcher, callback);
    }
};

}  // namespace

namespace ov::pass {

PagedCausalConv1DFusion::PagedCausalConv1DFusion() {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool PagedCausalConv1DFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PagedCausalConv1DFusion);

    ov::ParameterVector created_parameters;
    const auto track_created_parameter = [&created_parameters](const ParameterCreationResult& creation_result) {
        if (creation_result.created) {
            created_parameters.push_back(creation_result.parameter);
        }
        return creation_result.parameter;
    };

    SharedRuntimeInputs shared_inputs{
        track_created_parameter(
            create_or_get_named_parameter(model, "subsequence_begins", ov::element::i32, ov::PartialShape{-1})),
        track_created_parameter(create_or_get_named_parameter(model, "paged_conv_block_indices", ov::element::i32, ov::PartialShape{-1})),
        track_created_parameter(
            create_or_get_named_parameter(model, "paged_conv_block_indices_begins", ov::element::i32, ov::PartialShape{-1})),
        track_created_parameter(create_or_get_named_parameter(model, "paged_conv_past_lens", ov::element::i32, ov::PartialShape{-1})),
        track_created_parameter(
            create_or_get_named_parameter(model, "paged_conv_cache_interval", ov::element::i32, ov::PartialShape{-1}))};

    std::map<size_t, std::shared_ptr<v0::Parameter>> conv_state_tables;
    for (const auto& node : model->get_ordered_ops()) {
        for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
            const auto output = node->output(output_index);
            const auto layer_index = extract_conv_layer_index(output);
            if (!layer_index || conv_state_tables.count(*layer_index)) {
                continue;
            }

            const auto conv_state_table = create_or_get_named_parameter(model,
                                                                        make_conv_state_table_name(*layer_index),
                                                                        output.get_element_type(),
                                                                        make_conv_state_table_shape(output.get_partial_shape()));
            conv_state_tables[*layer_index] = conv_state_table.parameter;
            if (conv_state_table.created) {
                created_parameters.push_back(conv_state_table.parameter);
            }
        }
    }

    ov::pass::Manager manager(get_pass_config(), "PagedCausalConv1DFusion");
    manager.set_per_pass_validation(false);
    manager.register_pass<PagedCausalConv1DFusionMatcher>(shared_inputs, conv_state_tables);
    const bool rewritten = manager.run_passes(model);

    for (const auto& parameter : created_parameters) {
        if (parameter->output(0).get_target_inputs().empty() && model->get_parameter_index(parameter) >= 0) {
            model->remove_parameter(parameter);
        }
    }

    return rewritten;
}

}  // namespace ov::pass

