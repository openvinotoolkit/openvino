// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_data_movement.hpp"

#include "itt.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

std::vector<size_t> get_indices_by_op_type(const std::shared_ptr<Node>& main_node) {
    if (as_type_ptr<ov::op::util::PadBase>(main_node)) {
        return {1, 2};
    } else if (as_type_ptr<ov::op::v1::BatchToSpace>(main_node) || as_type_ptr<ov::op::v1::SpaceToBatch>(main_node)) {
        return {1, 2, 3};
    } else {
        return {};
    }
}

}  // namespace

TSDataMovementForward::TSDataMovementForward() {
    MATCHER_SCOPE(TSDataMovementForward);
    create_pattern<op::util::PadBase, ov::op::v1::BatchToSpace, ov::op::v1::SpaceToBatch, ov::op::v0::ReverseSequence>(
        {0});

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        // remove Transpose on 1st input:
        auto transpose_parent = main_node->input_value(0).get_node()->input_value(0);
        main_node->input(0).replace_source_output(transpose_parent);

        const auto transpose_axis_order = transpose_info.transpose_const->get_axis_vector_val();
        const auto reversed_transpose_order = ReverseTransposeOrder(transpose_axis_order);
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);

        const auto& indices = get_indices_by_op_type(main_node);
        for (const auto& idx : indices) {
            main_node->input(idx).replace_source_output(
                ChangeValuesOrder(main_node->input_value(idx), reversed_transpose_order, axis));
        }

        if (auto reverse_seq = as_type_ptr<ov::op::v0::ReverseSequence>(main_node)) {
            reverse_seq->set_batch_axis(transpose_axis_order[reverse_seq->get_batch_axis()]);
            reverse_seq->set_sequence_axis(transpose_axis_order[reverse_seq->get_sequence_axis()]);
        }

        default_outputs_update(main_node, transpose_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSDataMovementBackward::TSDataMovementBackward() {
    MATCHER_SCOPE(TSDataMovementBackward);

    auto main_node_label =
        wrap_type<op::util::PadBase, ov::op::v1::BatchToSpace, ov::op::v1::SpaceToBatch, ov::op::v0::ReverseSequence>(
            [](const Output<Node>& output) -> bool {
                return has_static_rank()(output) && CheckTransposeConsumers(output);
            });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                                                                       /* input_indexes= */ {0})) {
            register_new_node(new_node);
        }

        RemoveTransposeConsumers(main_node);
        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        const auto reversed_transpose_order = ReverseTransposeOrder(transpose_axis_order);
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        const auto& indices = get_indices_by_op_type(main_node);
        for (const auto& idx : indices) {
            main_node->input(idx).replace_source_output(
                ChangeValuesOrder(main_node->input_value(idx), transpose_axis_order, axis));
        }

        if (auto reverse_seq = as_type_ptr<ov::op::v0::ReverseSequence>(main_node)) {
            reverse_seq->set_batch_axis(reversed_transpose_order[reverse_seq->get_batch_axis()]);
            reverse_seq->set_sequence_axis(reversed_transpose_order[reverse_seq->get_sequence_axis()]);
        }
        main_node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
