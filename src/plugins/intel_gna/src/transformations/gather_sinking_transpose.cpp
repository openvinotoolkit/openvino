// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose.hpp"

#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;

namespace {

inline std::vector<std::shared_ptr<ov::Node>> merge_nodes_forward(std::shared_ptr<ov::Node> gather,
                                                                  std::shared_ptr<ov::Node> transpose) {
    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
    auto gather_ids = ov::as_type_ptr<Constant>(gather->get_input_node_shared_ptr(1));
    auto gather_axis = ov::as_type_ptr<Constant>(gather->get_input_node_shared_ptr(2));

    // transpose ids -> gather indexes
    const ov::AxisVector transpose_ids =
        graph_utils::make_gather_indexes_from_transpose_axes(transpose->get_input_shape(0),
                                                             transpose_const->get_axis_vector_val());
    // merge gather indexes
    const ov::AxisVector gather_new_ids =
        graph_utils::combine_gather_indexes(gather_ids->get_axis_vector_val(), transpose_ids);

    // new gather
    auto gather_new_const_ids =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_new_ids.size()}, gather_new_ids);
    auto gather_new = std::make_shared<Gather>(gather->input_value(0), gather_new_const_ids, gather_axis);

    ov::Shape shape_out = transpose->get_output_shape(0);
    auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
    auto reshape_out = std::make_shared<Reshape>(gather_new, reshape_out_const, false);

    replace_node_update_name(transpose, reshape_out);

    return std::vector<std::shared_ptr<ov::Node>>({gather_new, reshape_out});
}

inline std::vector<std::shared_ptr<ov::Node>> merge_nodes_backward(std::shared_ptr<ov::Node> gather,
                                                                   std::shared_ptr<ov::Node> transpose) {
    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
    auto gather_ids = ov::as_type_ptr<Constant>(gather->get_input_node_shared_ptr(1));
    auto gather_axis = ov::as_type_ptr<Constant>(gather->get_input_node_shared_ptr(2));

    ov::Shape shape_in = gather->get_input_shape(0);
    auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
    auto reshape_in = std::make_shared<Reshape>(transpose->input_value(0), reshape_in_const, false);

    // transpose ids -> gather indexes
    const ov::AxisVector transpose_ids =
        graph_utils::make_gather_indexes_from_transpose_axes(transpose->get_input_shape(0),
                                                             transpose_const->get_axis_vector_val());
    // merge gather indexes
    const ov::AxisVector gather_new_ids =
        graph_utils::combine_gather_indexes(gather_ids->get_axis_vector_val(), transpose_ids);

    // new gather
    auto gather_new_const_ids =
        std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_new_ids.size()}, gather_new_ids);
    auto gather_new = std::make_shared<Gather>(reshape_in, gather_new_const_ids, gather_axis);

    ov::replace_node_update_name(gather, gather_new);

    return std::vector<std::shared_ptr<ov::Node>>({reshape_in, gather_new});
}

inline bool is_skip_operation(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<Reshape>(node) != nullptr && node->output(0).get_target_inputs().size() == 1;
}

}  // namespace

GatherSinkingTransposeForward::GatherSinkingTransposeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeForward);
    auto gather_ids_label = wrap_type<Constant>(graph_utils::is_constant_1d);
    auto gather_label = wrap_type<Gather>({any_input(), gather_ids_label, any_input()});
    auto reshape_label = wrap_type<Reshape>({gather_label, any_input()});
    // auto transpose_label = wrap_type<Transpose>({reshape_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto reshape = as_type_ptr<Reshape>(pattern_to_output.at(reshape_label).get_node_shared_ptr());

        // skip all the Reshape layers
        std::shared_ptr<ov::Node> non_reshape_node =
            graph_utils::get_next_node_skipping_certain(reshape, is_skip_operation);
        auto transpose = std::dynamic_pointer_cast<Transpose>(non_reshape_node);
        if (!transpose) {
            return false;
        }

        const std::vector<std::shared_ptr<ov::Node>> new_nodes = merge_nodes_forward(gather, transpose);
        for (const auto& node : new_nodes) {
            register_new_node(node);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeBackward::GatherSinkingTransposeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeBackward);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()});
    auto gather_ids_label = wrap_type<Constant>(graph_utils::is_constant_1d);
    auto gather_label = wrap_type<Gather>({reshape_label, gather_ids_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto reshape = as_type_ptr<Reshape>(pattern_to_output.at(reshape_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());

        // skip all the Reshape layers
        std::shared_ptr<ov::Node> non_reshape_node =
            graph_utils::get_prev_node_skipping_certain(reshape, is_skip_operation);
        auto transpose = std::dynamic_pointer_cast<Transpose>(non_reshape_node);
        if (!transpose) {
            return false;
        }

        const std::vector<std::shared_ptr<ov::Node>> new_nodes = merge_nodes_backward(gather, transpose);
        for (const auto& node : new_nodes) {
            register_new_node(node);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

bool GatherSinkingTranspose::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(GatherSinkingTranspose);
    {
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<GatherSinkingTransposeForward>();
        manager.register_pass<GatherSinkingTransposeBackward>();
        manager.run_passes(model);
    }
    return false;
}
