// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"

namespace ov {
namespace op {
namespace v0 {

Result::Result(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void Result::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Argument has ", get_input_size(), " outputs (1 expected).");

    // Result doesn't change change in/out tensors
    auto& output = get_output_descriptor(0);
    auto& input = get_input_descriptor(0);
    output.set_tensor_ptr(input.get_tensor_ptr());
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Result_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<Result>(new_args.at(0));
}

bool Result::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Result_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1);

    if (outputs.empty()) {
        outputs.emplace_back(inputs[0].get_element_type(), inputs[0].get_shape());
    } else {
        OPENVINO_ASSERT(outputs.size() == 1);
        if (!outputs[0]) {
            outputs[0] = Tensor(inputs[0].get_element_type(), inputs[0].get_shape());
        }
    }

    outputs[0].set_shape(inputs[0].get_shape());
    if (inputs[0].get_element_type() == element::string) {
        // memcpy for element::string Tensor does not work because output elements
        // will refer to input string elements but they must be separate objects in memory
        inputs[0].copy_to(outputs[0]);
    } else {
        void* output = outputs[0].data();
        const void* input = inputs[0].data();
        memcpy(output, input, outputs[0].get_byte_size());
    }

    return true;
}

bool Result::has_evaluate() const {
    OV_OP_SCOPE(v0_Result_has_evaluate);
    return true;
}

bool Result::can_constant_fold(const OutputVector& input_values) const {
    return false;
}

ov::Layout Result::get_layout() const {
    return ov::layout::get_layout(output(0));
}

void Result::set_layout(const ov::Layout& layout) {
    ov::layout::set_layout(output(0), layout);
}
}  // namespace v0
}  // namespace op

AttributeAdapter<ResultVector>::AttributeAdapter(ResultVector& ref) : m_ref(ref) {}

bool AttributeAdapter<ResultVector>::visit_attributes(AttributeVisitor& visitor) {
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size()) {
        m_ref.resize(size);
    }
    std::ostringstream index;
    for (size_t i = 0; i < size; i++) {
        index.str("");
        index << i;
        std::string id;
        if (m_ref[i]) {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i]) {
            m_ref[i] = as_type_ptr<op::v0::Result>(visitor.get_registered_node(std::move(id)));
        }
    }
    return true;
}
}  // namespace ov
