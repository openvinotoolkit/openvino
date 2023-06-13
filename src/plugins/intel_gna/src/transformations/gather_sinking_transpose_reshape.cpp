// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include "openvino/cc/ngraph/itt.hpp"

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "log/debug.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::intel_gna;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::pass;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace gather_sinking;

namespace {
using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

NodePair sink_transpose_forward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    const auto gather_indices_value =
        make_gather_indices_from_transpose_axes(transpose->get_input_shape(0),
                                                transpose_constant->get_axis_vector_val());
    int64_t gather_axis_value = graph_utils::get_first_valuable_dim_id(reshape->get_output_shape(0));
    if (gather_axis_value < 0)
        gather_axis_value = 0;

    auto reshape_new = reshape->clone_with_new_inputs({transpose->input_value(0), reshape->input_value(1)});

    auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, gather_axis_value);
    auto gather_indices =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape_new, gather_indices, gather_axis);

    ov::replace_node(reshape, gather);

    ov::copy_runtime_info({reshape}, {gather, gather_indices, gather_axis, reshape_new});
    gather->set_friendly_name(reshape->get_friendly_name());

    return std::make_pair(reshape_new, gather);
}

NodePair sink_transpose_backward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    int64_t gather_axis_value = graph_utils::get_first_valuable_dim_id(reshape->get_input_shape(0));
    if (gather_axis_value < 0)
        gather_axis_value = 0;
    const auto gather_indices_value =
        make_gather_indices_from_transpose_axes(transpose->get_input_shape(0),
                                                transpose_constant->get_axis_vector_val());

    auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, gather_axis_value);
    auto gather_indices =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape->input_value(0), gather_indices, gather_axis);

    auto reshape_const_new = std::make_shared<Constant>(ov::element::i64,
                                                        ov::Shape{transpose->get_output_shape(0).size()},
                                                        transpose->get_output_shape(0));
    auto reshape_new = std::make_shared<Reshape>(gather, reshape_const_new, false);

    ov::replace_node(transpose, reshape_new);

    ov::copy_runtime_info({transpose}, {gather, gather_indices, gather_axis, reshape_new, reshape_const_new});
    reshape_new->set_friendly_name(transpose->get_friendly_name());

    return std::make_pair(gather, reshape_new);
}

bool are_flatten_shapes(const ov::Shape& shape1, const ov::Shape& shape2) {
    size_t i = 0;
    // find non-equal parts
    while (shape1[i] == shape2[i]) {
        ++i;
    }
    // consider only last dimension to be flatten/unflatten
    if (shape1.size() - 1 != i && shape2.size() - 1 != i)
        return false;
    // min_shape.back() == MULTIPLY(max_shape.begin() + i, max_shape.end())
    const size_t mult1 = std::accumulate(shape1.begin() + i, shape1.end(), 1, std::multiplies<size_t>());
    const size_t mult2 = std::accumulate(shape2.begin() + i, shape2.end(), 1, std::multiplies<size_t>());
    return mult1 == mult2;
}

bool is_tail_flatten(const ov::Output<ov::Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const ov::Shape input_shape = graph_utils::trim_shape(reshape_node->get_input_shape(0));
    const ov::Shape output_shape = graph_utils::trim_shape(reshape_node->get_output_shape(0));
    return input_shape.size() > output_shape.size() && are_flatten_shapes(input_shape, output_shape);
}

bool is_tail_unflatten(const ov::Output<ov::Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const ov::Shape input_shape = graph_utils::trim_shape(reshape_node->get_input_shape(0));
    const ov::Shape output_shape = graph_utils::trim_shape(reshape_node->get_output_shape(0));
    return input_shape.size() < output_shape.size() && are_flatten_shapes(input_shape, output_shape);
}

bool is_transpose_unsupported(const ov::Output<ov::Node>& output) {
    return !Limitations::is_transpose_supported(output.get_node_shared_ptr());
}

bool is_backward_sinking_enables(const ov::Output<ov::Node>& output) {
    return is_transpose_unsupported(output) && ov::is_sinking_node(output.get_node_shared_ptr());
}

}  // namespace

// working with situation when we transpose dims that are flatten/unflatten
// consider only if flatten/unflatten are last dimensions
GatherSinkingTransposeReshapeForward::GatherSinkingTransposeReshapeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeForward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label}, is_transpose_unsupported);
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()}, is_tail_flatten);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const =
            ov::as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = graph_utils::trim_shape(reshape->get_shape());
        const ov::Shape transpose_shape = graph_utils::trim_shape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::remove_single_input_node(transpose);
            return true;
        }

        const NodePair new_nodes = sink_transpose_forward(transpose, transpose_const, reshape);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        update_forward_gather_sinking_ability(new_nodes.second);
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeReshapeBackward::GatherSinkingTransposeReshapeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeBackward);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()}, is_tail_unflatten);
    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({reshape_label, transpose_const_label}, is_backward_sinking_enables);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = graph_utils::trim_shape(reshape->get_input_shape(0));
        const ov::Shape transpose_shape = graph_utils::trim_shape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::remove_single_input_node(transpose);
            return true;
        }

        const NodePair new_nodes = sink_transpose_backward(transpose, transpose_const, reshape);
        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
