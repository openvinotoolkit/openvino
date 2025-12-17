#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "../utils.hpp"



namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_smooth_l1_loss(const NodeContext& context) {
    num_inputs_check(context, 2, 4);

    auto input = context.get_input(0);
    auto target = context.get_input(1);


    float beta_val = context.get_input_size() > 2
                         ? context.const_input<float>(2)
                         : 1.0f;


    int64_t reduction_mode = context.get_input_size() > 3
                                 ? context.const_input<int64_t>(3)
                                 : 1; // 0: none, 1: mean, 2: sum

    auto dtype = input.get_element_type();

    auto diff = context.mark_node(
        std::make_shared<v1::Subtract>(input, target));

    auto abs_diff = context.mark_node(
        std::make_shared<v0::Abs>(diff));

    auto beta_const = context.mark_node(
        v0::Constant::create(dtype, Shape{}, {beta_val}));

    auto half_const = context.mark_node(
        v0::Constant::create(dtype, Shape{}, {0.5f}));

    auto half_beta_const = context.mark_node(
        v0::Constant::create(dtype, Shape{}, {0.5f * beta_val}));

    auto condition = context.mark_node(
        std::make_shared<v1::Less>(abs_diff, beta_const));

    auto squared_diff = context.mark_node(
        std::make_shared<v1::Multiply>(abs_diff, abs_diff));

    auto div_l2 = context.mark_node(
        std::make_shared<v1::Divide>(squared_diff, beta_const));

    auto l2_branch = context.mark_node(
        std::make_shared<v1::Multiply>(half_const, div_l2));

    auto l1_branch = context.mark_node(
        std::make_shared<v1::Subtract>(abs_diff, half_beta_const));

    Output<Node> loss = context.mark_node(
        std::make_shared<v1::Select>(condition, l2_branch, l1_branch));

    if (reduction_mode == 1 || reduction_mode == 2) {
        auto axes = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {-1}));

        if (reduction_mode == 1) {
            loss = context.mark_node(
                std::make_shared<v1::ReduceMean>(loss, axes, false));
        } else {
            loss = context.mark_node(
                std::make_shared<v1::ReduceSum>(loss, axes, false));
        }
    }

    return {loss};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
