#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_set_(const NodeContext& context) {
    num_inputs_check(context, 2, 2, true);

    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    if (lhs.get_element_type() != rhs.get_element_type()) {
        FRONT_END_GENERAL_CHECK(false,
                                "Cannot set tensor of type ",
                                lhs.get_element_type().to_string(),
                                " with values from tensor of type ",
                                rhs.get_element_type().to_string());
    }

    context.mutate_input(0, rhs);
    return {rhs};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
