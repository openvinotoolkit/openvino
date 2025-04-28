#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_nested_tensor_from_mask(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    auto tensor = context.get_input(0);
    auto mask = context.get_input(1);

    auto mask_bool = mask;
    
    if(mask.get_element_type() != element::boolen){
        mask_bool = context.mark_node(std::make_shared<ov::op::v1::Convert>(mask, element::boolean))->output(0);
    }
    
    auto masked_tensor = context.mark_node(std::make_shared<ov::op::v1::Select>(
        mask_bool, tensor, 
        context.mark_node(std::make_shared<ov::op::v0::Constant>(
            tensor.get_element_type(), tensor.get_shape(), 0))->output(0)))->output(0);
    
    auto mask_sum = context.mark_node(std::make_shared<ov::op::v4::ReduceSum>(
        context.mark_node(std::make_shared<ov::op::v1::Convert>(mask_bool, element::i32))->output(0),
        context.mark_node(std::make_shared<ov::op::v0::Constant>(
            element::i32, Shape{1}, std::vector<int32_t>{1}))->output(0),
        true))->output(0);
    
    
    auto metadata = mask_sum;
    
    
    if (!context.input_is_none(2)) {
        context.mutate_input(2, masked_tensor);
    }
    
    return {masked_tensor, metadata};
    
}
    
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
