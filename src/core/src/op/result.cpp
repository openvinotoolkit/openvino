// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Result);

op::Result::Result(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

op::Result::Result(const Output<Node>& arg, bool) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Result::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Result_visit_attributes);
    return true;
}

void op::Result::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Argument has ", get_input_size(), " outputs (1 expected).");

    // Result doesn't change change in/out tensors
    auto& output = get_output_descriptor(0);
    auto& input = get_input_descriptor(0);
    output.set_tensor_ptr(input.get_tensor_ptr());
}

shared_ptr<Node> op::Result::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Result_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto res = make_shared<Result>(new_args.at(0));
    return std::move(res);
}

bool op::Result::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Result_evaluate);
    outputs[0]->set_unary(inputs[0]);
    void* output = outputs[0]->get_data_ptr();
    void* input = inputs[0]->get_data_ptr();
    memcpy(output, input, outputs[0]->get_size_in_bytes());

    return true;
}

bool op::Result::has_evaluate() const {
    OV_OP_SCOPE(v0_Result_has_evaluate);
    return true;
}

bool op::Result::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}

ov::Layout op::Result::get_layout() const {
    return ov::layout::get_layout(output(0));
}

void op::Result::set_layout(const ov::Layout& layout) {
    ov::layout::set_layout(output(0), layout);
}

BWDCMP_RTTI_DEFINITION(ov::AttributeAdapter<ResultVector>);

ov::AttributeAdapter<ResultVector>::AttributeAdapter(ResultVector& ref) : m_ref(ref) {}

bool ov::AttributeAdapter<ResultVector>::visit_attributes(AttributeVisitor& visitor) {
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size()) {
        m_ref.resize(size);
    }
    ostringstream index;
    for (size_t i = 0; i < size; i++) {
        index.str("");
        index << i;
        string id;
        if (m_ref[i]) {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i]) {
            m_ref[i] = ov::as_type_ptr<ngraph::op::v0::Result>(visitor.get_registered_node(id));
        }
    }
    return true;
}
