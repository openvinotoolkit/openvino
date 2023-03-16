// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

std::vector<float> get_floats(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape) {
    size_t input_size = ngraph::shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case ngraph::element::Type_t::bf16: {
        ngraph::bfloat16* p = input->get_data_ptr<ngraph::bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ngraph::element::Type_t::f16: {
        ngraph::float16* p = input->get_data_ptr<ngraph::float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ngraph::element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape) {
    size_t input_size = ngraph::shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case ngraph::element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
        break;
    }

    return result;
}

namespace {
template <ngraph::element::Type_t ET>
bool evaluate(std::shared_ptr<ngraph::Node> op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    return false;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::Relu>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        ngraph::shape_size(inputs[0]->get_shape()));
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::Assign>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::ReadValue>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <typename T>
bool evaluate_node(std::shared_ptr<ngraph::Node> node,
                   const ngraph::HostTensorVector& outputs,
                   const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<T>(node), outputs, inputs);
    default:
        throw ngraph::ngraph_error(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                                   std::string("in evaluate_node()"));
    }
}
}  // namespace

ngraph::runtime::interpreter::EvaluatorsMap& ngraph::runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return evaluatorsMap;
}
