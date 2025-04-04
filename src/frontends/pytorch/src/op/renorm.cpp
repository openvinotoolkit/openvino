#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_renorm(const NodeContext& context) {
    // aten::renorm(Tensor self, float p, int dim, float maxnorm) -> Tensor
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto p_val = get_input_with_floating_type(context, 1);
    int32_t dim_val = context.const_input<int32_t>(2);
    dim_val = dim_val ^ 1;
    auto maxnorm_val = get_input_with_floating_type(context, 3);
    auto abs_input = context.mark_node(std::make_shared<v0::Abs>(input));
    auto axis_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {dim_val}));
    auto one_const = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));

    // norm = (sum(|input|^p, dim, keepdim=True))^(1/p)
    auto powered = context.mark_node(std::make_shared<v1::Power>(abs_input, p_val));
    auto sum_p = context.mark_node(std::make_shared<v1::ReduceSum>(powered, axis_const, true));
    auto inv_p_const = context.mark_node(std::make_shared<v1::Divide>(one_const, p_val));
    auto norm = context.mark_node(std::make_shared<v1::Power>(sum_p, inv_p_const));

    // scale = min(1, maxnorm / norm)
    auto scale = context.mark_node(std::make_shared<v1::Divide>(maxnorm_val, norm));
    scale = context.mark_node(std::make_shared<v1::Minimum>(one_const, scale));
    auto output = context.mark_node(std::make_shared<v1::Multiply>(input, scale));

    return {output};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov