#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/clamp.hpp"
#include "utils_quantize.hpp"
namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;
OutputVector translate_quantized_relu6(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto x = context.get_input(0);
    const auto relu6 = context.mark_node(std::make_shared<v0::Clamp>(x, 0.0, 6.0));
    const auto quantized_pt_node = cast_quantized_fw_node(x.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(quantized_pt_node, "Input must be a quantized tensor");
    return {quantize(context, relu6, quantized_pt_node->get_scale(), quantized_pt_node->get_zero_point(), x)};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov