#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_renorm(const NodeContext& context) {
    // aten::renorm(Tensor self, float p, int dim, float maxnorm) -> Tensor
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    float p_val = context.const_input<float>(1);
    int64_t dim_val = context.const_input<int64_t>(2);
    float maxnorm_val = context.const_input<float>(3);

    // Compute the absolute value of the input.
    auto abs_input = context.mark_node(std::make_shared<v0::Abs>(input));

    // Create a constant for the reduction axis along which we compute the norm.
    auto axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {dim_val}));

    // Compute the norm of the input along the given dimension.
    Output<Node> norm;
    if (std::isinf(p_val)) {
        // When p is infinite, use ReduceMax for +infinity norm and ReduceMin for -infinity norm.
        if (p_val > 0) {
            norm = context.mark_node(std::make_shared<v1::ReduceMax>(abs_input, axis_const, true));
        } else {
            norm = context.mark_node(std::make_shared<v1::ReduceMin>(abs_input, axis_const, true));
        }
    } else {
        // norm = (sum(|input|^p, dim, keepdim=True))^(1/p)
        auto p_const = context.mark_node(v0::Constant::create(input.get_element_type(), Shape{}, {p_val}));
        auto powered = context.mark_node(std::make_shared<v1::Power>(abs_input, p_const));
        auto sum_p = context.mark_node(std::make_shared<v1::ReduceSum>(powered, axis_const, true));
        auto inv_p_const = context.mark_node(v0::Constant::create(input.get_element_type(), Shape{}, {1.0f / p_val}));
        norm = context.mark_node(std::make_shared<v1::Power>(sum_p, inv_p_const));
    }

    // scale = min(1, maxnorm / norm)
    auto maxnorm_const = context.mark_node(v0::Constant::create(input.get_element_type(), Shape{}, {maxnorm_val}));
    auto one_const = context.mark_node(v0::Constant::create(input.get_element_type(), Shape{}, {1.0f}));
    auto ratio = context.mark_node(std::make_shared<v1::Divide>(maxnorm_const, norm));
    auto scale = context.mark_node(std::make_shared<v1::Minimum>(one_const, ratio));
    auto output = context.mark_node(std::make_shared<v1::Multiply>(input, scale));

    return {output};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov