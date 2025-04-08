#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov{
namespace frontend{
namespace pytorch{
namespace op{

using namespace ov::op;

OutputVector translate_to_copy(const NodeContext& context) {
    num_inputs_check(context, 5, 6);

    Output<Node> input = context.get_input(0);


    if (context.input_is_none(1)) {

        return {input};
    }

    Output<Node> dtype_node = context.get_input(1);
    auto dtype_const = std::dynamic_pointer_cast<v0::Constant>(dtype_node.get_node_shared_ptr());

    PYTORCH_OP_CONVERSION_CHECK(dtype_const, "Expected constant dtype for aten::_to_copy");


    auto dtype_value = dtype_const->cast_vector<int64_t>()[0];
    auto target_type = convert_dtype(dtype_value);

    auto result = context.mark_node(std::make_shared<v0::Convert>(input, target_type));
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov