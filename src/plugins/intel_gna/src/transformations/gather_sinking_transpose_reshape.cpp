// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "log/debug.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::limitations;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace gather_sinking;

namespace {

std::vector<std::shared_ptr<ov::Node>> gather_sink_forward(std::shared_ptr<ov::Node> transpose,
                                                           std::shared_ptr<ov::Node> reshape) {
    std::vector<std::shared_ptr<ov::Node>> new_nodes = {};

    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
    const auto gather_indexes_value =
        graph_utils::make_gather_indexes_from_transpose_axes(transpose->get_input_shape(0),
                                                             transpose_const->get_axis_vector_val());
    ov::Shape shape_out = reshape->get_output_shape(0);

    // Set Gather input shape
    std::vector<int8_t> reshape_in_dims = {1, -1};
    for (size_t i = reshape_in_dims.size(); i < shape_out.size(); ++i) {
        if (shape_out[i] == 1) {
            reshape_in_dims.push_back(1);
        }
    }
    // reshape
    auto reshape_in_const =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{reshape_in_dims.size()}, reshape_in_dims);
    auto reshape_in = reshape->clone_with_new_inputs({transpose->input_value(0), reshape_in_const});
    new_nodes.push_back(reshape_in);

    const int64_t gather_axis_value = graph_utils::get_first_valuable_dim_id(reshape_in->get_output_shape(0));
    auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, gather_axis_value);
    auto gather_indices =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indexes_value.size()}, gather_indexes_value);
    auto gather = std::make_shared<Gather>(reshape_in, gather_indices, gather_axis);
    new_nodes.push_back(gather);

    auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
    auto reshape_out = std::make_shared<Reshape>(gather, reshape_out_const, false);

    if (!graph_utils::are_shapes_equal(reshape_out->get_input_shape(0), reshape_out->get_output_shape(0))) {
        new_nodes.push_back(reshape_out);
    }

    ov::replace_output_update_name(reshape->output(0), new_nodes.back()->output(0));
    ov::copy_runtime_info({reshape}, new_nodes);

    return new_nodes;
}

std::vector<std::shared_ptr<ov::Node>> gather_sink_backward(std::shared_ptr<ov::Node> reshape,
                                                            std::shared_ptr<ov::Node> transpose) {
    std::vector<std::shared_ptr<ov::Node>> new_nodes = {};
    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
    ov::Shape shape_out = transpose->get_output_shape(0);

    // Set Gather input shape
    std::vector<int8_t> reshape_in_dims = {1, -1};
    for (size_t i = reshape_in_dims.size(); i < shape_out.size(); ++i) {
        if (shape_out[i] == 1) {
            reshape_in_dims.push_back(1);
        }
    }

    // reshape
    auto reshape_in_const =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{reshape_in_dims.size()}, reshape_in_dims);
    auto reshape_in = reshape->clone_with_new_inputs({reshape->input_value(0), reshape_in_const});
    new_nodes.push_back(reshape_in);

    const int64_t gather_axis_value = graph_utils::get_first_valuable_dim_id(reshape_in->get_output_shape(0));
    const auto gather_indexes_value =
        graph_utils::make_gather_indexes_from_transpose_axes(transpose->get_input_shape(0),
                                                             transpose_const->get_axis_vector_val());

    auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, gather_axis_value);
    auto gather_indices =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indexes_value.size()}, gather_indexes_value);
    auto gather = std::make_shared<Gather>(reshape_in, gather_indices, gather_axis);
    new_nodes.push_back(gather);

    auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
    auto reshape_out = std::make_shared<Reshape>(gather, reshape_out_const, false);

    if (!graph_utils::are_shapes_equal(reshape_out->get_input_shape(0), reshape_out->get_output_shape(0))) {
        new_nodes.push_back(reshape_out);
    }

    ov::replace_node_update_name(transpose, new_nodes.back());
    ov::copy_runtime_info({transpose}, new_nodes);

    return new_nodes;
}

bool is_transpose_unsupported(const ov::Output<ov::Node>& output) {
    return !Limitations::is_transpose_supported(output.get_node_shared_ptr());
}

bool is_backward_sinking_enabled(const ov::Output<ov::Node>& output) {
    return is_transpose_unsupported(output) && ov::is_sinking_node(output.get_node_shared_ptr());
}

}  // namespace

GatherSinkingTransposeReshapeForward::GatherSinkingTransposeReshapeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeForward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label}, is_transpose_unsupported);
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = graph_utils::trim_shape(reshape->get_shape());
        const ov::Shape transpose_shape = graph_utils::trim_shape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::remove_single_input_node(transpose);
            return true;
        }

        const std::vector<std::shared_ptr<ov::Node>> new_nodes = gather_sink_forward(transpose, reshape);

        for (const auto& node : new_nodes) {
            register_new_node(node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeReshapeBackward::GatherSinkingTransposeReshapeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeBackward);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()});
    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({reshape_label, transpose_const_label}, is_backward_sinking_enabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = graph_utils::trim_shape(reshape->get_input_shape(0));
        const ov::Shape transpose_shape = graph_utils::trim_shape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::remove_single_input_node(transpose);
            return true;
        }

        const std::vector<std::shared_ptr<ov::Node>> new_nodes = gather_sink_backward(reshape, transpose);
        for (const auto& node : new_nodes) {
            register_new_node(node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
