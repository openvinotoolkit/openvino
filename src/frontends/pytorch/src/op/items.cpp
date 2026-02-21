#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_items(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto dict_input = context.get_input(0);
    auto dict_node = dict_input.get_node_shared_ptr();

    auto fw_node = cast_fw_node(dict_node, "prim::DictConstruct");
    PYTORCH_OP_CONVERSION_CHECK(fw_node,
                                "aten::items conversion supports only dictionaries created via prim::DictConstruct");

    const auto& inputs = fw_node->input_values();

    PYTORCH_OP_CONVERSION_CHECK(inputs.size() % 2 == 0,
                                "prim::DictConstruct must have even number of inputs (key-value pairs)");

    OutputVector tuple_outputs;

    for (size_t i = 0; i < inputs.size(); i += 2) {
        auto key = inputs[i];
        auto value = inputs[i + 1];

        // Create (key, value) tuple using prim::TupleConstruct
        auto tuple_node = context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(),
                                                                              OutputVector{key, value},
                                                                              1,
                                                                              "prim::TupleConstruct"));

        tuple_outputs.push_back(tuple_node);
    }

    // Create list of tuples using prim::ListConstruct
    auto list_node = context.mark_node(
        std::make_shared<PtFrameworkNode>(context.get_decoder(), tuple_outputs, 1, "prim::ListConstruct"));

    return {list_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
