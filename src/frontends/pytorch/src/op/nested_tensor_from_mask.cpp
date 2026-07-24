#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate__nested_tensor_from_mask(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto input = context.get_input(0);
    auto mask = context.get_input(1);
    auto mask_i32 = context.mark_node(std::make_shared<v0::Convert>(mask, element::i32));
    auto flat_mask_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flat_mask = context.mark_node(std::make_shared<v1::Reshape>(mask_i32, flat_mask_shape, false));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto last_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_neg_one, const_zero));
    auto flat_input_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{flat_mask_shape, last_dim}, 0));
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, flat_input_shape, true));
    auto non_zero = context.mark_node(std::make_shared<v3::NonZero>(flat_mask));
    auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    auto transposed = context.mark_node(std::make_shared<v1::Transpose>(non_zero, input_order));
    auto squeeze_axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto row_idx = context.mark_node(std::make_shared<v0::Squeeze>(transposed, squeeze_axes));
    row_idx = context.mark_node(std::make_shared<v0::Convert>(row_idx, element::i32));
    auto axis_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto flat_tokens = context.mark_node(std::make_shared<v8::Gather>(flat_input, row_idx, axis_0));
    return {flat_tokens};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
