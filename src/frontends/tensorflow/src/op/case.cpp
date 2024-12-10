#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/translate_session.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

using namespace ov::opsets;

OutputVector translate_case_op(const NodeContext& node) {
    // Validate the operation type
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, op_type == "Case",
                             "Internal error: incorrect usage of translate_case_op.");

    // Retrieve the number of branches and inputs
    auto num_branches = node.get_attribute<int>("branches");
    TENSORFLOW_OP_VALIDATION(node, num_branches > 0,
                             "[TensorFlow Frontend] Case operation must have at least one branch.");

    // The first input is the condition for selecting the branch
    auto cond = node.get_input(0);

    // Create a list to store sub-graphs for the branches
    std::vector<std::shared_ptr<Model>> branch_graphs;
    for (int i = 0; i < num_branches; ++i) {
        std::string branch_name = "branch_" + std::to_string(i);
        auto branch_body = node.get_attribute<std::string>(branch_name);
        
        // Ensure that the branch model is correctly loaded
        auto branch_model = node.get_translate_session()->get_body_ov_model(branch_body, node.get_inputs());
        TENSORFLOW_OP_VALIDATION(node, branch_model, 
                                 "[TensorFlow Frontend] Failed to retrieve body graph for branch: " + branch_name);
        branch_graphs.push_back(branch_model);
    }

    // Create the nested If operation to represent the Case operation
    std::shared_ptr<Model> current_model = nullptr;
    for (int i = num_branches - 1; i >= 0; --i) {
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(branch_graphs[i]);

        if (current_model) {
            if_op->set_else_body(current_model);
        } else {
            // Default empty else body
            auto placeholder_model = std::make_shared<Model>(OutputVector{}, ParameterVector{});
            if_op->set_else_body(placeholder_model);
        }

        current_model = if_op->get_body_model();
    }

    // Set the outputs and names
    auto outputs = current_model->get_results();
    OutputVector ov_outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs[i]->output(0).get_tensor();
        tensor.set_names({node.get_name() + ":" + std::to_string(i)});
        ov_outputs.push_back(outputs[i]->output(0));
    }

    return ov_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
