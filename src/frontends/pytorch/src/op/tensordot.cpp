#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_tensordot(const NodeContext& context) {
    num_inputs_check(context, 3, 4);

    auto a = context.get_input(0);
    auto b = context.get_input(1);
    auto matmul_result = context.mark_node(std::make_shared<v0::MatMul>(a, b, false, false));

    if (!context.input_is_none(2)) {
        auto dims_a = context.get_input(2);
        auto reduce_sum_result = context.mark_node(std::make_shared<v1::ReduceSum>(matmul_result, dims_a, true));
        return {reduce_sum_result};
    }

    return {matmul_result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
