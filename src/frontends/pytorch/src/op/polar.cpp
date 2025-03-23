#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_polar(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto abs = context.get_input(0);
    auto angle = context.get_input(1);
    auto real = context.mark_node(std::make_shared<v1::Multiply>(abs,context.mark_node(std::make_shared<v0::Cos>(angle))));
    auto imag = context.mark_node(std::make_shared<v1::Multiply>(abs,context.mark_node(std::make_shared<v0::Sin>(angle))));
    auto complex_concat = context.mark_node(std::make_shared<v0::Concat>(OutputVector{real, imag}, -1));
    // wrap the tensor with ComplexTypeMark to flag it as complex for later operations.
    auto complex_tensor = context.mark_node(std::make_shared<ComplexTypeMark>(complex_concat));
    return {complex_tensor};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov