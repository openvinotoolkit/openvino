#include "openvino/core/constant_fold_utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/pass/constant_folding.hpp"

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
    auto node = output.get_node_shared_ptr();
    return (!node->has_evaluate() || ov::is_type<ov::op::v0::Constant>(node)) &&
           is_type_unsupported(output.get_element_type());
}

bool ov::util::is_convert(const std::shared_ptr<Node>& node) {
    return ov::is_type<op::v0::Convert>(node) || ov::is_type<op::v1::ConvertLike>(node);
}

std::shared_ptr<ov::Node> ov::util::try_clone_and_convert_inputs(const std::shared_ptr<ov::Node>& node) {
    if (ov::pass::constant_folding_is_disabled(node))
        return node;

    if (is_convert(node))
        return node;

    bool requires_conversion = false;
    size_t num_inputs = node->get_input_size();
    for (size_t i = 0; i < num_inputs; i++) {
        if (!ov::is_type<ov::op::v0::Constant>(node->get_input_node_ptr(i))) {
            // non-constant input - node is not constfoldable
            return node;
        }
        if (node_evaluate_requires_input_conversion(node->input(i))) {
            // ith input is a constant but it requires a convert to f32
            requires_conversion = true;
        }
    }

    if (!requires_conversion)
        return node;

    auto inputs = node->input_values();
    for (size_t i = 0; i < num_inputs; i++) {
        if (node_evaluate_requires_input_conversion(node->input(i))) {
            // this input is a constant, but it requires conversion from unsupported type to f32
            auto convert = std::make_shared<ov::op::v0::Convert>(inputs[i], ov::element::f32);
            OutputVector outputs(1);
            OPENVINO_ASSERT(convert->constant_fold(outputs, convert->input_values()));
            inputs[i] = outputs[0];
        }
    }

    // Create a new node with new (converted) inputs.
    // After validate_and_infer_types - cloned node should be constfoldable.
    auto cloned_node = node->clone_with_new_inputs(inputs);

    // But it may not be constfoldable if `cloned_node` output element types
    // don't depend on input element types - if that's the case -
    // following function: `node_evaluate_requires_output_conversion` returns true.
    //
    // Unfortunately, this piece of code doesn't support such cases (yet). So if you ever see
    // the error below - check what types make the `node_evaluate_requires_output_conversion`
    // return true and add those types to evaluate function for this particular operator.
    for (size_t i = 0; i < cloned_node->get_output_size(); i++) {
        OPENVINO_ASSERT(!node_evaluate_requires_output_conversion(cloned_node->output(i)),
                        cloned_node,
                        " is not constfoldable");
    }

    return cloned_node;
}
