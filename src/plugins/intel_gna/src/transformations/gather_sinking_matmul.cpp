// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_matmul.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;

namespace {
bool Has2dInputs(const Output<Node>& output) {
    auto node = output.get_node_shared_ptr();
    auto input_left_rank = node->get_input_partial_shape(0).rank();
    auto input_right_rank = node->get_input_partial_shape(0).rank();
    return (input_left_rank.is_static() && input_right_rank.is_static() &&
            input_left_rank.get_length() == 2 && input_right_rank.get_length() == 2);
}

bool HasGatherInputs(const Output<Node>& output) {
    return !GetFirstGatherInput(output.get_node_shared_ptr()).isEmpty();
}

bool IsSinked(const Output<Node>& output) {
    return Has2dInputs(output) && HasGatherInputs(output);
}

int64_t Swap2DNegativeAxis(int64_t axis) {
    if (axis == -1)
        return -2;
    return -1;
}

size_t GetAnotherMatMulIndex(size_t input_idx) {
    if (!input_idx)
        return 1;
    return 0;
}

bool IsMatMulInputTransposed(const std::shared_ptr<MatMul>& matmul, size_t input_idx) {
    if (!input_idx)
        return matmul->get_transpose_a();
    return matmul->get_transpose_b();
}

} // namespace

GatherSinkingMatmulForward::GatherSinkingMatmulForward() {
    MATCHER_SCOPE(GatherSinkingMatmulForward);

    auto matmul_label = wrap_type<MatMul>({any_input(), any_input()}, IsSinked);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto matmul = as_type_ptr<MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
        GatherInputsInfo gather_input_info = GetFirstGatherInput(matmul);

        int64_t gather_negative_axis =
        GetNormalizedNegativeGatherAxis(gather_input_info.axis_const,
                                        gather_input_info.gather->get_input_partial_shape(0).rank().get_length());
        gather_negative_axis = Swap2DNegativeAxis(gather_negative_axis);
        if (IsMatMulInputTransposed(matmul, GetAnotherMatMulIndex(gather_input_info.input_idx)))
            gather_negative_axis = Swap2DNegativeAxis(gather_negative_axis);

        sink_forward::UpdateInputGather(matmul, gather_input_info, &gather_negative_axis);
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
