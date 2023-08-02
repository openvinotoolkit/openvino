// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

#include "ops/ops_evaluates.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
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

std::vector<int64_t> get_signal_size(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs,
                                     size_t num_of_axes) {
    if (inputs.size() == 3) {
        return get_integers(inputs[2], inputs[2]->get_shape());
    }

    return std::vector<int64_t>(num_of_axes, static_cast<int64_t>(-1));
}

ngraph::runtime::interpreter::EvaluatorsMap& ngraph::runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return evaluatorsMap;
}
