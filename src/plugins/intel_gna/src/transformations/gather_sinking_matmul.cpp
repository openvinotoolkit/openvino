// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_matmul.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;
using namespace ov::intel_gna::pass::helper;

namespace {
bool has_gather_inputs(const Output<Node>& output) {
    return !get_first_gather_input(output.get_node_shared_ptr()).isEmpty();
}

bool is_matmul_sinked(const Output<Node>& output) {
    return has_2d_inputs(output) && has_gather_inputs(output);
}

int64_t get_another_input_negative_axis(int64_t axis) {
    if (axis == -1)
        return -2;
    return -1;
}

}  // namespace

GatherSinkingMatmulForward::GatherSinkingMatmulForward() {
    MATCHER_SCOPE(GatherSinkingMatmulForward);

    auto matmul_label = wrap_type<MatMul>({any_input(), any_input()}, is_matmul_sinked);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto matmul = as_type_ptr<MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
        GatherInputsInfo gather_input_info = get_first_gather_input(matmul);

        int64_t gather_negative_axis = get_normalized_negative_gather_axis(
            gather_input_info.axis_const,
            gather_input_info.gather->get_input_partial_shape(0).rank().get_length());
        gather_negative_axis = get_another_input_negative_axis(gather_negative_axis);
        const auto another_input_idx = gather_input_info.input_idx ? 0 : 1;
        if (is_matmul_input_transposed(matmul, another_input_idx))
            gather_negative_axis = get_another_input_negative_axis(gather_negative_axis);

        sink_forward::update_input_gather(matmul, gather_input_info, &gather_negative_axis);
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingMatmulBackward::GatherSinkingMatmulBackward() {
    MATCHER_SCOPE(GatherSinkingMatmulBackward);

    auto matmul_label = wrap_type<MatMul>({any_input(), any_input()}, has_2d_inputs);
    auto indices_const_label = wrap_type<Constant>(is_constant_1d);
    auto axis_const_label = wrap_type<Constant>(is_constant_1d);
    auto gather_label = wrap_type<Gather>({matmul_label, indices_const_label, axis_const_label},
                                          [](const Output<Node>& output) -> bool {
                                              return has_static_rank()(output) && is_gather_sinking_node(output);
                                          });

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto matmul = as_type_ptr<MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
        auto axis_const = as_type_ptr<Constant>(pattern_to_output.at(axis_const_label).get_node_shared_ptr());
        auto indices_const = as_type_ptr<Constant>(pattern_to_output.at(indices_const_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());

        int64_t gather_negative_axis =
            get_normalized_negative_gather_axis(axis_const, gather->get_input_partial_shape(0).rank().get_length());
        const int matmul_insert_input_idx = static_cast<int>(
            convert_axis_to_positive(gather_negative_axis, gather->get_input_partial_shape(0).rank().get_length()));
        if (is_matmul_input_transposed(matmul, static_cast<size_t>(gather_negative_axis)))
            gather_negative_axis = get_another_input_negative_axis(gather_negative_axis);
        auto new_axis_const = std::make_shared<Constant>(axis_const->get_element_type(), Shape{}, gather_negative_axis);
        for (auto& new_node : sink_backward::insert_gather_before_node(matmul,
                                                                       indices_const,
                                                                       new_axis_const,
                                                                       gather,
                                                                       std::vector<int>{matmul_insert_input_idx})) {
            register_new_node(new_node);
        }

        // remove output transposes
        remove_single_output_consumers(matmul);

        swap_names(gather, matmul);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
