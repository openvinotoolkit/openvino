#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/pytorch/node_context.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_extend(const NodeContext& context) {
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() == 2,
                                  "aten::extend expects 2 inputs: the list and the iterable to extend with.");
    auto list_to_extend_ov = context.get_input(0);
    auto iterable_to_add_ov = context.get_input(1);

    auto const_list1 = ov::as_type_ptr<ov::op::v0::Constant>(list_to_extend_ov.get_node_shared_ptr());
    auto const_list2 = ov::as_type_ptr<ov::op::v0::Constant>(iterable_to_add_ov.get_node_shared_ptr());

    FRONT_END_OP_CONVERSION_CHECK(const_list1 && const_list2,
                                  "Translation for aten::extend currently requires both inputs "
                                  "to be constant tensors (representing lists constructed from constants).");

    int64_t concatenation_axis = 0;
    ov::OutputVector inputs_to_concat = {list_to_extend_ov, iterable_to_add_ov};

    auto extend_result_node =
        context.mark_node(std::make_shared<ov::op::v0::Concat>(inputs_to_concat, concatenation_axis));

    Output<Node> final_ov_node = extend_result_node;
    OutputVector folded_result;

    if (extend_result_node->has_evaluate()) {
        folded_result.resize(extend_result_node->get_output_size());
        if (extend_result_node->constant_fold(folded_result, {list_to_extend_ov, iterable_to_add_ov})) {
            final_ov_node = folded_result[0];
        }
    }

    context.mutate_input(0, final_ov_node);

    return {final_ov_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov