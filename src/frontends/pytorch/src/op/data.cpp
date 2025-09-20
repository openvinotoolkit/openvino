
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_data(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);

    if (context.get_decoder()->get_output_type(0).is<type::Complex>()) {
        auto mark = std::make_shared<ComplexTypeMark>(input, input.get_element_type());
        return {context.mark_node(mark)};
    }

    return {input};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
