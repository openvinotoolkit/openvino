// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_linspace_op(const NodeContext& node) {
    /* LinSpace operation can be expressed in the following form:
     * 1) Compute deltas by which each element will be increased in each slice
     * through new dimension as delta = (stop - start) / (num - 1)
     * 2) Generate a range of numbers by which times delta will be added to start to
     * compute new elements in each slide as range = [0, 1, ..., num - 1]
     * 3) Unsqueeze start and delta by new axis. And unsqueeze range by axes except given axis
     * 4) Compute the result of the operation as result = start + delta * range
     */
    auto num_inputs = node.get_input_size();
    auto start = node.get_input(0);
    auto stop = node.get_input(1);
    auto num = node.get_input(2);

    TENSORFLOW_OP_VALIDATION(node, start.get_partial_shape().rank().is_static(), "Input rank must be static.");
    int64_t start_rank = start.get_partial_shape().rank().get_length();

    // retrieve axis from Constant node and compute a range of axes except given axis
    // for unsqueezing start and delta tensors
    std::vector<int64_t> axis;
    std::vector<int64_t> except_axis_range;
    if (num_inputs > 3 && start_rank > 0) {
        get_const_input(node, 3, &axis);
        TENSORFLOW_OP_VALIDATION(node, axis.size() == 1, "Axis must be a scalar for LinSpace operation.");
        axis[0] = axis[0] >= 0 ? axis[0] : start_rank + 1 + axis[0];
        for (int64_t dim_ind = 0; dim_ind < start_rank + 1; ++dim_ind) {
            if (dim_ind != axis[0]) {
                except_axis_range.push_back(dim_ind);
            }
        }
    }

    TENSORFLOW_OP_VALIDATION(node,
                             axis.empty() && start_rank == 0 || axis.size() == 1 && start_rank > 0,
                             "Axis must be used only if input for LinSpace operation is ND tensor.");

    auto one = make_shared<Constant>(num.get_element_type(), Shape{}, 1);
    auto num_minus_1 = make_shared<ConvertLike>(make_shared<Subtract>(num, one), start);
    auto delta = make_shared<Divide>(make_shared<Subtract>(stop, start), num_minus_1);

    auto zero = make_shared<Constant>(num.get_element_type(), Shape{}, 0);
    auto range_0_num_minus_1 =
        make_shared<ConvertLike>(make_shared<Range>(zero, num, one, num.get_element_type()), start);

    // convert a case with scalar inputs
    if (axis.empty() && start_rank == 0) {
        auto delta_mul_range = make_shared<Multiply>(delta, range_0_num_minus_1);
        auto result = make_shared<Add>(start, delta_mul_range);
        set_node_name(node.get_name(), result);
        return result->outputs();
    }

    auto const_axis = make_shared<Constant>(element::i64, Shape{axis.size()}, axis);
    auto const_except_axis = make_shared<Constant>(element::i64, Shape{except_axis_range.size()}, except_axis_range);

    auto unsqueeze_start = make_shared<Unsqueeze>(start, const_axis);
    auto unsqueeze_delta = make_shared<Unsqueeze>(delta, const_axis);
    auto unsqueeze_range = make_shared<Unsqueeze>(range_0_num_minus_1, const_except_axis);

    auto delta_mul_range = make_shared<Multiply>(unsqueeze_delta, unsqueeze_range);
    auto result = make_shared<Add>(unsqueeze_start, delta_mul_range);
    set_node_name(node.get_name(), result);
    return result->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
