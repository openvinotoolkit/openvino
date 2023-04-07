// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_concat.hpp"

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSConcatForward::TSConcatForward() {
    MATCHER_SCOPE(TSConcatForward);

    auto main_node_label = wrap_type<Concat>(IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);
        auto concat_node = as_type_ptr<Concat>(main_node);
        auto concat_axis = concat_node->get_concatenation_axis();
        if (concat_axis < 0) {
            return false;
        }
        // todo: support dyn rank case
        bool updated = sink_forward::UpdateInputTransposes(main_node, transpose_input_info);
        if (!updated) {
            return false;
        }

        const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        const int64_t transposed_concat_axis = transpose_axis_order[concat_axis];
        concat_node->set_axis(transposed_concat_axis);
        concat_node->set_concatenation_axis(-1);

        main_node->validate_and_infer_types();
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
            UpdateForwardSinkingAbility(new_node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSConcatBackward::TSConcatBackward() {
    MATCHER_SCOPE(TSConcatBackward);

    auto main_node_label = wrap_type<Concat>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && HasSameOutputTransposeNodes(output);
    });

    auto transpose_const_label = wrap_type<Constant>();

    auto transpose_label =
        wrap_type<Transpose>({main_node_label, transpose_const_label}, [](const Output<Node>& output) -> bool {
            return has_static_rank()(output) && is_sinking_node(output);
        });

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        auto concat_node = as_type_ptr<Concat>(main_node);
        auto concat_axis = concat_node->get_concatenation_axis();
        if (concat_axis < 0) {
            return false;
        }

        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        const auto reversed_transpose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        if (static_cast<int64_t>(reversed_transpose_axis_order.size()) <= concat_axis) {
            return false;
        }

        const auto transposed_concat_axis = reversed_transpose_axis_order[concat_axis];
        concat_node->set_axis(static_cast<int64_t>(transposed_concat_axis));
        concat_node->set_concatenation_axis(-1);

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_const)) {
            register_new_node(new_node);
        }
        concat_node->validate_and_infer_types();

        RemoveSingleOutputConsumers(main_node);
        SwapNames(transpose, main_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
