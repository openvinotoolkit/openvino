// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_cumsum.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

#undef CUMSUM_AXIS_INPUT_IDX
#define CUMSUM_AXIS_INPUT_IDX 1

TSCumSumForward::TSCumSumForward() {
    MATCHER_SCOPE(TSCumSumForward);

    create_pattern<ov::op::v0::CumSum>({0});

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        if (transformation_callback(main_node)) {
            return false;
        }

        bool res = utils::sink_forward::UpdateInputTransposes(main_node, transpose_info, /* input_indexes= */ {0});
        if (!res)
            return res;

        const auto transpose_axis_order = transpose_info.transpose_const->get_axis_vector_val();
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        const auto& new_axes = ChangeAxes(main_node->input_value(CUMSUM_AXIS_INPUT_IDX), transpose_axis_order, axis);
        main_node->input(CUMSUM_AXIS_INPUT_IDX).replace_source_output(new_axes);

        default_outputs_update(main_node, transpose_info);
        return true;
    };
    transpose_sinking(matcher_name, sinking_transformation);
}

TSCumSumBackward::TSCumSumBackward() {
    MATCHER_SCOPE(TSCumSumBackward);
    auto main_node_label = wrap_type<ov::op::v0::CumSum>([](const Output<Node>& output) -> bool {
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
        auto new_axes = ChangeAxes(main_node->input_value(CUMSUM_AXIS_INPUT_IDX), reversed_transpose_order, axis);
        main_node->input(CUMSUM_AXIS_INPUT_IDX).replace_source_output(new_axes);

        main_node->validate_and_infer_types();
        return true;
    };
    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
