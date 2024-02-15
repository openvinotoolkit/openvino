#include "openvino/core/constant_fold_utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/fake_convert.hpp"
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

bool ov::util::is_type_unsupported(const ov::element::Type& type) {
    const auto& unsupported_types = ov::util::unsupported_types();
    return std::find(unsupported_types.begin(), unsupported_types.end(), type) != unsupported_types.end();
}

namespace {

template <typename... Args>
struct IsAnyOfType;

template <>
struct IsAnyOfType<> {
    static bool is_any_of_type(const ov::Node* const node) {
        return false;
    }
};

template <typename T, typename... Args>
struct IsAnyOfType<T, Args...> {
    static bool is_any_of_type(const ov::Node* const node) {
        return ov::is_type<T>(node) || IsAnyOfType<Args...>::is_any_of_type(node);
    }
};

template <typename... Args>
bool is_any_of_type(const ov::Node* const node) {
    return IsAnyOfType<Args...>::is_any_of_type(node);
}

}  // namespace

bool ov::util::node_requires_precision_conversion(const ov::Node* const node) {
    if (node->get_input_size() == 0 || node->get_output_size() == 0) {
        return false;
    }

#define WHITELIST                                                                                    \
    op::util::AssignBase, v0::Ceiling, v0::Constant, v0::Convert, v1::ConvertLike, v13::FakeConvert, \
        op::util::MultiSubGraphOp, op::util::ReadValueBase
    // any node that is on WHITELIST does not require precision conversion
    using namespace ov::op;
    if (is_any_of_type<WHITELIST>(node)) {
        return false;
    }
#undef WHITELIST

    bool has_unsupported_type = false;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (ov::util::is_type_unsupported(node->get_input_element_type(i))) {
            has_unsupported_type = true;
            break;
        }
    }
    if (!has_unsupported_type) {
        for (size_t i = 0; i < node->get_output_size(); i++) {
            if (ov::util::is_type_unsupported(node->get_output_element_type(i))) {
                has_unsupported_type = true;
            }
        }
    }

    return has_unsupported_type && node->has_evaluate();
}

static bool convert_range_precision(const std::shared_ptr<ov::Node>& node) {
    auto range = ov::as_type_ptr<ov::op::v4::Range>(node);
    if (!range)
        return false;

    if (ov::util::is_type_unsupported(range->get_output_type())) {
        range->set_output_type(ov::element::f32);
        return true;
    }
    return false;
}

static const std::unordered_map<ov::NodeTypeInfo, std::function<bool(const std::shared_ptr<ov::Node>&)>>
    output_conversion_methods = {
        {ov::op::v4::Range::get_type_info_static(), convert_range_precision},
};

std::shared_ptr<ov::Node> ov::util::convert_to_supported_precision(const Node* const node) {
    return ov::util::convert_to_supported_precision(node, node->input_values());
}

std::shared_ptr<ov::Node> ov::util::convert_to_supported_precision(const Node* const node, const OutputVector& inputs) {
    size_t num_inputs = node->get_input_size();
    OutputVector converted_inputs;
    converted_inputs.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        const auto& input_type = inputs[i].get_element_type();
        if (ov::util::is_type_unsupported(input_type)) {
            auto convert = std::make_shared<ov::op::v0::Convert>(inputs[i], ov::element::f32);
            OutputVector replacements(1);
            if (convert->constant_fold(replacements, convert->input_values())) {
                converted_inputs.push_back(replacements[0]);
            } else {
                converted_inputs.push_back(convert);
            }
        } else {
            converted_inputs.push_back(inputs[i]);
        }
    }

    // Create a new node with new (converted) inputs.
    auto cloned_node = node->clone_with_new_inputs(converted_inputs);

    // Override TypeRelaxed types
    auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(cloned_node);
    if (type_relaxed) {
        for (size_t i = 0; i < num_inputs; i++) {
            if (ov::util::is_type_unsupported(type_relaxed->get_origin_input_type(i))) {
                type_relaxed->set_origin_input_type(cloned_node->get_input_element_type(i), i);
            }
        }
        for (size_t i = 0; i < cloned_node->get_output_size(); i++) {
            if (ov::util::is_type_unsupported(cloned_node->get_output_element_type(i))) {
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

static void convert_tensor(const ov::Tensor& input, ov::Tensor& output) {
    auto outputs = ov::TensorVector{output};
    OPENVINO_ASSERT(ov::op::v0::Convert().evaluate(outputs, ov::TensorVector{input}),
                    "unable to convert tensor with type ",
                    input.get_element_type(),
                    " to ",
                    output.get_element_type());
}

bool ov::util::evaluate_node_with_unsupported_precision(const ov::Node* node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs) {
    auto create_node_from_tensors = [](const Node* const node, const TensorVector& inputs) {
        OutputVector new_inputs;
        new_inputs.reserve(inputs.size());
        for (const auto& input : inputs)
            new_inputs.push_back(std::make_shared<op::v0::Constant>(input));
        return node->clone_with_new_inputs(new_inputs);
    };

    // Handle case when node inputs don't match input tensors.
    // Some use cases:
    // - node without inputs which its only purpose is to run evaluate, e.g. Convert().evaluate({output}, {input})
    // - type relaxed nodes that may have its input types overriden during evaluate
    bool input_tensors_match_node_inputs = [node, inputs]() -> bool {
        size_t node_input_size = node->get_input_size();
        size_t num_tensors = inputs.size();
        if (num_tensors > node_input_size) {
            return false;
        }
        for (size_t i = 0; i < num_tensors; i++) {
            if (node->get_input_element_type(i) != inputs[i].get_element_type()) {
                return false;
            }
        }
        return true;
    }();

    std::shared_ptr<Node> node_shared_ptr;
    if (!input_tensors_match_node_inputs) {
        node_shared_ptr = create_node_from_tensors(node, inputs);
        node = node_shared_ptr.get();
    }

    if (!ov::util::node_requires_precision_conversion(node)) {
        return false;
    }

    auto cloned = ov::util::convert_to_supported_precision(node);

    TensorVector converted_input_tensors;
    converted_input_tensors.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        // convert input tensors to f32 if this input tensor's type is unsupported
        if (ov::util::is_type_unsupported(inputs[i].get_element_type())) {
            converted_input_tensors.emplace_back(element::f32, inputs[i].get_shape());
            convert_tensor(inputs[i], converted_input_tensors.back());
        } else {
            converted_input_tensors.push_back(inputs[i]);
        }
    }

    TensorVector converted_output_tensors;
    converted_output_tensors.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        if (cloned->get_output_element_type(i) != outputs[i].get_element_type()) {
            converted_output_tensors.emplace_back(cloned->get_output_element_type(i), outputs[i].get_shape());
        } else {
            converted_output_tensors.push_back(outputs[i]);
        }
    }

    // evaluate converted node
    if (!cloned->evaluate(converted_output_tensors, converted_input_tensors)) {
        return false;
    }

    // convert outputs tensors from f32 to original type if necessary
    for (size_t i = 0; i < outputs.size(); i++) {
        if (converted_output_tensors[i].get_element_type() != outputs[i].get_element_type()) {
            convert_tensor(converted_output_tensors[i], outputs[i]);
        }
    }
    return true;
}
