// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose.hpp"

#include <utility>

#include "openvino/cc/ngraph/itt.hpp"
// #include "transformations/utils/utils.hpp"
#include "common/graph_utils.hpp"
#include "openvino/opsets/opset10.hpp"
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
        graph_utils::make_gather_indices_from_transpose_axes(transpose->get_input_shape(0),
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
        graph_utils::make_gather_indices_from_transpose_axes(transpose->get_input_shape(0),
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

inline bool is_constant_1d(const ov::Output<ov::Node>& output) {
    return output.get_partial_shape().size() <= 1;
}

}  // namespace

GatherSinkingTransposeForward::GatherSinkingTransposeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeForward);
    auto gather_ids_label = wrap_type<Constant>(is_constant_1d);
    auto gather_label = wrap_type<Gather>({any_input(), gather_ids_label, any_input()});
    auto reshape_label = wrap_type<Reshape>({gather_label, any_input()});
    auto transpose_label = wrap_type<Transpose>({reshape_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather_ids = as_type_ptr<Constant>(pattern_to_output.at(gather_ids_label).get_node_shared_ptr());
        // auto gather_axis = as_type_ptr<Constant>(pattern_to_output.at(gather_axis_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());

        // auto transpose_const =
        // as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label).get_node_shared_ptr());

        const std::vector<std::shared_ptr<ov::Node>> new_nodes = merge_nodes_forward(gather, transpose);
        for (const auto& node : new_nodes) {
            register_new_node(node);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeBackward::GatherSinkingTransposeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeBackward);

    auto transpose_label = wrap_type<Transpose>({any_input(), any_input()});
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()});
    auto gather_ids_label = wrap_type<Constant>(is_constant_1d);
    auto gather_label = wrap_type<Gather>({reshape_label, gather_ids_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label).get_node_shared_ptr());

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
