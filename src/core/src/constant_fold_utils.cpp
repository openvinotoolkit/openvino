#include "openvino/core/constant_fold_utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/reference/convert.hpp"
#include "ov_ops/type_relaxed.hpp"

const ov::element::TypeVector& ov::util::unsupported_types() {
    static const ov::element::TypeVector types{ov::element::f16, ov::element::bf16};
    return types;
}

static bool is_type_unsupported(const ov::element::Type& type) {
    const auto& unsupported_types = ov::util::unsupported_types();
    return std::find(unsupported_types.begin(), unsupported_types.end(), type) != unsupported_types.end();
}

bool ov::util::node_requires_precision_conversion(const std::shared_ptr<const ov::Node>& node) {
    if (node->get_input_size() == 0)
        return false;
    if (node->has_evaluate()) {
        return false;
    }
    // ReadValue or Assign cannot be cloned with new inputs with different precision
    if (ov::is_type<ov::op::util::ReadValueBase>(node) || ov::is_type<ov::op::util::AssignBase>(node)) {
        return false;
    }
    // ConvertLike has constant_fold function but doesn't have has_evaluate function
    if (ov::is_type<ov::op::v1::ConvertLike>(node)) {
        return false;
    }
    if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        return false;
    }
    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (is_type_unsupported(node->get_input_element_type(i))) {
            return true;
        }
    }
    for (size_t i = 0; i < node->get_output_size(); i++) {
        if (is_type_unsupported(node->get_output_element_type(i))) {
            return true;
        }
    }
    return false;
}

static bool convert_range_precision(const std::shared_ptr<ov::Node>& node) {
    auto range = ov::as_type_ptr<ov::op::v4::Range>(node);
    if (!range)
        return false;

    if (is_type_unsupported(range->get_output_type())) {
        range->set_output_type(ov::element::f32);
        return true;
    }
    return false;
}

static std::unordered_map<ov::NodeTypeInfo, std::function<bool(const std::shared_ptr<ov::Node>&)>>
    output_conversion_methods = {
        {ov::op::v4::Range::get_type_info_static(), convert_range_precision},
};

std::shared_ptr<ov::Node> ov::util::convert_to_supported_precision(const std::shared_ptr<ov::Node>& node) {
    return ov::util::convert_to_supported_precision(node, node->input_values());
}

std::shared_ptr<ov::Node> ov::util::convert_to_supported_precision(const std::shared_ptr<ov::Node>& node,
                                                                   OutputVector&& inputs) {
    size_t num_inputs = node->get_input_size();
    if (num_inputs == 0)
        return node;

    if (ov::is_type<ov::op::util::MultiSubGraphOp>(node))
        return node;

    bool requires_conversion = false;
    bool inputs_changed = false;

    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (is_type_unsupported(inputs[i].get_element_type())) {
            requires_conversion = true;
        }
        inputs_changed = inputs_changed || node->input_value(i) != inputs[i];
    }

    for (size_t i = 0; i < node->get_output_size(); i++) {
        if (is_type_unsupported(node->get_output_element_type(i))) {
            requires_conversion = true;
            break;
        }
    }

    if (!requires_conversion) {
        if (inputs_changed) {
            return node->clone_with_new_inputs(inputs);
        }
        return node;
    }

    for (size_t i = 0; i < num_inputs; i++) {
        const auto& input_type = inputs[i].get_element_type();
        if (is_type_unsupported(input_type)) {
            auto convert = std::make_shared<ov::op::v0::Convert>(inputs[i], ov::element::f32);
            OutputVector replacements(1);
            if (convert->constant_fold(replacements, convert->input_values())) {
                inputs[i] = replacements[0];
            } else {
                inputs[i] = convert;
            }
        }
    }

    // Create a new node with new (converted) inputs.
    // After validate_and_infer_types - cloned node should be constfoldable.
    auto cloned_node = node->clone_with_new_inputs(inputs);

    // Override TypeRelaxed types
    auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(cloned_node);
    if (type_relaxed) {
        for (size_t i = 0; i < cloned_node->get_input_size(); i++) {
            if (is_type_unsupported(type_relaxed->get_origin_input_type(i))) {
                type_relaxed->set_origin_input_type(cloned_node->get_input_element_type(i), i);
            }
        }
        for (size_t i = 0; i < cloned_node->get_output_size(); i++) {
            if (is_type_unsupported(cloned_node->get_output_element_type(i))) {
                type_relaxed->set_overridden_output_type(element::f32, i);
            }
        }
        cloned_node->validate_and_infer_types();
    }

    // Handle nodes which outputs precisions don't depend on input precisions
    auto method_it = output_conversion_methods.find(cloned_node->get_type_info());
    if (method_it != output_conversion_methods.end()) {
        if (method_it->second(cloned_node)) {
            cloned_node->validate_and_infer_types();
        }
    }

    return cloned_node;
}

bool ov::util::constant_fold_node(const std::shared_ptr<Node>& node, OutputVector& output_constants) {
    auto converted = node;
    if (ov::util::node_requires_precision_conversion(node)) {
        converted = convert_to_supported_precision(node);
    }

    auto num_outputs = converted->get_output_size();
    if (output_constants.size() < num_outputs)
        output_constants.resize(num_outputs);

    if (!converted->constant_fold(output_constants, converted->input_values()))
        return false;

    for (size_t i = 0; i < num_outputs; i++) {
        if (output_constants[i].get_element_type() == node->get_output_element_type(i))
            continue;
        auto convert = std::make_shared<ov::op::v0::Convert>(output_constants[i], node->get_output_element_type(i));
        OutputVector convert_output(1);
        OPENVINO_ASSERT(convert->constant_fold(convert_output, convert->input_values()));
        output_constants[i] = convert_output[0];
    }
    return true;
}

static void convert_tensor(const ov::Tensor& input, ov::Tensor& output) {
    auto outputs = ov::TensorVector{output};
    OPENVINO_ASSERT(ov::op::v0::Convert().evaluate(outputs, ov::TensorVector{input}),
                    "unable to convert tensor with type ",
                    input.get_element_type(),
                    " to ",
                    output.get_element_type());
}

bool ov::util::evaluate_node(const std::shared_ptr<ov::Node>& node,
                             const ov::TensorVector& input_tensors,
                             ov::TensorVector& output_tensors,
                             const ov::EvaluationContext& evaluation_context) {
    // create output_tensors if necessary
    if (output_tensors.size() < node->get_output_size()) {
        output_tensors.reserve(node->get_output_size());
        for (size_t i = output_tensors.size(); i < node->get_output_size(); i++) {
            output_tensors.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
        }
    }

    // if node can be evaluated as is - we're done
    if (node->evaluate(output_tensors, input_tensors, evaluation_context)) {
        return true;
    }
    // otherwise try to convert input tensors and run the Node::evaluate once again

    auto cloned = convert_to_supported_precision(node->shared_from_this());

    ov::TensorVector converted_input_tensors;
    converted_input_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); i++) {
        // convert input tensors to f32 if this input tensor's type is unsupported
        if (is_type_unsupported(input_tensors[i].get_element_type())) {
            converted_input_tensors.emplace_back(ov::element::f32, input_tensors[i].get_shape());
            convert_tensor(input_tensors[i], converted_input_tensors.back());
        } else {
            converted_input_tensors.push_back(input_tensors[i]);
        }
    }

    ov::TensorVector converted_output_tensors;
    converted_output_tensors.reserve(output_tensors.size());
    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (cloned->get_output_element_type(i) != output_tensors[i].get_element_type()) {
            converted_output_tensors.emplace_back(cloned->get_output_element_type(i), output_tensors[i].get_shape());
        } else {
            converted_output_tensors.push_back(output_tensors[i]);
        }
    }

    // evaluate converted node
    if (!cloned->evaluate(converted_output_tensors, converted_input_tensors, evaluation_context)) {
        return false;
    }

    // convert outputs tensors from f32 to original type if necessary
    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (converted_output_tensors[i].get_element_type() != output_tensors[i].get_element_type()) {
            convert_tensor(converted_output_tensors[i], output_tensors[i]);
        }
    }

    return true;
}
