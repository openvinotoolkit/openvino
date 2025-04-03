#include "openvino/op/reverse.hpp"
#include "utils.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"  // Include NodeContext definition
#include "openvino/src/frontends/pytorch/src/utils.hpp"
//#include "../../utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reverse(const NodeContext& context) {
    num_inputs_check(context, 2, 2);  // expects 2 inputs: tensor and dims
    auto x = context.get_input(0);
    auto dims = context.const_input<std::vector<int64_t>>(1);
    
    auto axes = ov::op::v0::Constant::create(
        element::i64, Shape{dims.size()}, dims
    );

    auto mode = ov::op::v1::Reverse::Mode::INDEX;
    return {std::make_shared<ov::op::v1::Reverse>(x, axes, mode)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov