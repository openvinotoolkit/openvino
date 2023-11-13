// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"

using namespace ov;

op::v0::Result::Result(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool op::v0::Result::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Result_visit_attributes);
    return true;
}

void op::v0::Result::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Argument has ", get_input_size(), " outputs (1 expected).");

    // Result doesn't change change in/out tensors
    auto& output = get_output_descriptor(0);
    auto& input = get_input_descriptor(0);
    output.set_tensor_ptr(input.get_tensor_ptr());
}

std::shared_ptr<Node> op::v0::Result::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Result_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto res = std::make_shared<Result>(new_args.at(0));
    return std::move(res);
}

bool op::v0::Result::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Result_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1);
    if (outputs.empty())
        outputs.emplace_back(ov::Tensor(inputs[0].get_element_type(), inputs[0].get_shape()));
    else
        OPENVINO_ASSERT(outputs.size() == 1);
    if (!outputs[0])
        outputs[0] = ov::Tensor(inputs[0].get_element_type(), inputs[0].get_shape());
    if (inputs[0].get_shape() != outputs[0].get_shape())
        outputs[0].set_shape(inputs[0].get_shape());
    void* output = outputs[0].data();
    void* input = inputs[0].data();
    memcpy(output, input, outputs[0].get_byte_size());

    return true;
}

bool op::v0::Result::has_evaluate() const {
    OV_OP_SCOPE(v0_Result_has_evaluate);
    return true;
}

bool op::v0::Result::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}

ov::Layout op::v0::Result::get_layout() const {
    return ov::layout::get_layout(output(0));
}

void op::v0::Result::set_layout(const ov::Layout& layout) {
    ov::layout::set_layout(output(0), layout);
}

ov::AttributeAdapter<ResultVector>::AttributeAdapter(ResultVector& ref) : m_ref(ref) {}

bool ov::AttributeAdapter<ResultVector>::visit_attributes(AttributeVisitor& visitor) {
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
            m_ref[i] = ov::as_type_ptr<ov::op::v0::Result>(visitor.get_registered_node(id));
        }
    }
    return true;
}
