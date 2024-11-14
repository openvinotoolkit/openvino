#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_masked_select(const NodeContext& context) {
    // aten::masked_select(Tensor self, Tensor mask, Tensor source) -> Tensor
    num_inputs_check(context, 2, 2);
    auto data = context.get_input(0);
    auto mask = context.get_input(1);
    auto res = masked_select(context, data, mask);
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
