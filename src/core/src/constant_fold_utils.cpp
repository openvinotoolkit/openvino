#include "openvino/core/constant_fold_utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/range.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/reference/convert.hpp"
#include "ov_ops/type_relaxed.hpp"

ov::element::TypeVector ov::util::unsupported_types() {
    return {ov::element::f16, ov::element::bf16};
}

static bool is_type_unsupported(const ov::element::Type& type) {
    auto types = ov::util::unsupported_types();
    return std::find(types.begin(), types.end(), type) != types.end();
}

/// \brief Checks if node has inputs which the node's evaluate doesn't support - those
///        inputs require conversion to a supported type.
///
/// \param node
///
/// \return true if node's evaluate doesn't support the node's inputs and false otherwise
static bool node_evaluate_requires_input_conversion(const ov::Input<ov::Node>& input) {
    return !input.get_node()->has_evaluate() && is_type_unsupported(input.get_element_type());
}

/// \brief Checks if node has outputs which the node's evaluate doesn't support
///
/// \param node
///
/// \return true if node's evaluate doesn't support the node's outputs and false otherwise
static bool node_evaluate_requires_output_conversion(const ov::Output<ov::Node>& output) {
    auto node = output.get_node();
    return (!node->has_evaluate() || ov::is_type<ov::op::v0::Constant>(node)) &&
           is_type_unsupported(output.get_element_type());
}

bool ov::util::is_convert(const std::shared_ptr<Node>& node) {
    return ov::is_type<op::v0::Convert>(node) || ov::is_type<op::v1::ConvertLike>(node);
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

std::shared_ptr<ov::Node> ov::util::try_convert_inputs(const std::shared_ptr<ov::Node>& node,
                                                       bool constant_fold_inputs) {
    return try_convert_inputs(node, node->input_values(), constant_fold_inputs);
}

std::shared_ptr<ov::Node> ov::util::try_convert_inputs(const std::shared_ptr<ov::Node>& node,
                                                       OutputVector inputs,
                                                       bool constant_fold_inputs) {
    size_t num_inputs = node->get_input_size();
    if (num_inputs == 0)
        return node;

    if (is_convert(node))
        return node;

    bool requires_conversion = false;
    bool inputs_changed = false;

    for (size_t i = 0; i < num_inputs; i++) {
        if (constant_fold_inputs && !ov::is_type<ov::op::v0::Constant>(inputs[i].get_node())) {
            // non-constant input - node is not constfoldable
            return node;
        }
        if (node->get_input_element_type(i) == inputs[i].get_element_type() &&
            node_evaluate_requires_input_conversion(node->input(i))) {
            // ith input requires a convert to f32
            requires_conversion = true;
        }
        inputs_changed = inputs_changed || node->input_value(i) != inputs[i];
    }

    size_t num_outputs = node->get_output_size();
    for (size_t i = 0; i < num_outputs; i++) {
        if (node_evaluate_requires_output_conversion(node->output(i))) {
            requires_conversion = true;
        }
    }

    if (!requires_conversion) {
        if (inputs_changed) {
            return node->clone_with_new_inputs(inputs);
        }
        return node;
    }

    for (size_t i = 0; i < num_inputs; i++) {
        if (node->get_input_element_type(i) == inputs[i].get_element_type() &&
            node_evaluate_requires_input_conversion(node->input(i))) {
            // this input is a constant, but it requires conversion from unsupported type to f32
            auto convert = std::make_shared<ov::op::v0::Convert>(inputs[i], ov::element::f32);
            if (constant_fold_inputs) {
                OutputVector outputs(1);
                OPENVINO_ASSERT(convert->constant_fold(outputs, convert->input_values()));
                inputs[i] = outputs[0];
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
            type_relaxed->set_origin_input_type(cloned_node->get_input_element_type(i), i);
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

    for (size_t i = 0; i < num_outputs; i++) {
        // The last check to make sure that 'cloned_node' outputs has supported precisions.
        // If they don't - it (most likely) means that the outputs precisions don't depend
        // in the node's inputs. In that case, there's an attribute in the operator that
        // specificies the output precision. In order to fix that - `output_conversion_methods`
        // map should be updated to handle that operator.
        OPENVINO_ASSERT(!node_evaluate_requires_output_conversion(cloned_node->output(i)),
                        cloned_node,
                        " output ",
                        i,
                        " has unsupported precision");
    }

    return cloned_node;
}

bool ov::util::constant_fold_node(const std::shared_ptr<Node>& node, OutputVector& output_constants) {
    auto cloned = try_convert_inputs(node);

    auto num_outputs = cloned->get_output_size();
    if (output_constants.size() < num_outputs)
        output_constants.resize(num_outputs);

    bool status = cloned->constant_fold(output_constants, cloned->input_values());
    if (!status)
        return status;

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

template <typename T>
static void convert_tensor(const ov::Tensor& input, ov::Tensor& output) {
    const auto& output_type = output.get_element_type();
    switch (output_type) {
    case ov::element::bf16: {
        ov::reference::convert(input.data<T>(), output.data<ov::bfloat16>(), ov::shape_size(input.get_shape()));
        return;
    }
    case ov::element::f16: {
        ov::reference::convert(input.data<T>(), output.data<ov::float16>(), ov::shape_size(input.get_shape()));
        return;
    }
    case ov::element::f32: {
        ov::reference::convert(input.data<T>(), output.data<float>(), ov::shape_size(input.get_shape()));
        return;
    }
    default:
        OPENVINO_THROW("unable to convert tensor with type ", input.get_element_type(), " to ", output_type);
    }
}

static void convert_tensor(const ov::Tensor& input, ov::Tensor& output) {
    switch (input.get_element_type()) {
    case ov::element::f16:
        convert_tensor<ov::float16>(input, output);
        return;
    case ov::element::bf16:
        convert_tensor<ov::bfloat16>(input, output);
        return;
    case ov::element::f32:
        convert_tensor<float>(input, output);
        return;
    default:
        OPENVINO_THROW("unable to convert tensor with type ",
                       input.get_element_type(),
                       " to ",
                       output.get_element_type());
    }
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

    // create a cloned node with converted inputs in order to know output precisions
    bool constant_fold_inputs = false;
    auto cloned = try_convert_inputs(node->shared_from_this(), constant_fold_inputs);

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
