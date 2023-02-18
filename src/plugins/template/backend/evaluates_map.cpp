// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/generate_proposal.hpp"
#include "ov_ops/augru_cell.hpp"
// #include "ov_ops/augru_sequence.hpp"

using namespace ngraph;
using namespace std;
namespace {
template <element::Type_t ET>
bool evaluate(shared_ptr<Node> op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    return false;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::Assign>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::ReadValue>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <typename T>
bool evaluate_node(std::shared_ptr<Node> node, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<op::v1::Select>(node) || ov::is_type<op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case element::Type_t::boolean:
        return evaluate<element::Type_t::boolean>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::bf16:
        return evaluate<element::Type_t::bf16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f16:
        return evaluate<element::Type_t::f16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f64:
        return evaluate<element::Type_t::f64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f32:
        return evaluate<element::Type_t::f32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i4:
        return evaluate<element::Type_t::i4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i8:
        return evaluate<element::Type_t::i8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i16:
        return evaluate<element::Type_t::i16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i32:
        return evaluate<element::Type_t::i32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i64:
        return evaluate<element::Type_t::i64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u1:
        return evaluate<element::Type_t::u1>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u4:
        return evaluate<element::Type_t::u4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u8:
        return evaluate<element::Type_t::u8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u16:
        return evaluate<element::Type_t::u16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u32:
        return evaluate<element::Type_t::u32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u64:
        return evaluate<element::Type_t::u64>(ov::as_type_ptr<T>(node), outputs, inputs);
    default:
        throw ngraph_error(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                           std::string("in evaluate_node()"));
    }
}
}  // namespace

runtime::interpreter::EvaluatorsMap& runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return evaluatorsMap;
}
