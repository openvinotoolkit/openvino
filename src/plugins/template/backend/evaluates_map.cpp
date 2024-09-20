// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

#include "ops/ops_evaluates.hpp"

std::vector<float> get_floats(const ov::Tensor& input, const ov::Shape& shape) {
    size_t input_size = ov::shape_size(shape);
    std::vector<float> result(input_size);

    switch (input.get_element_type()) {
    case ov::element::bf16: {
        ov::bfloat16* p = input.data<ov::bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ov::element::f16: {
        ov::float16* p = input.data<ov::float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ov::element::f32: {
        float* p = input.data<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const ov::Tensor& input, const ov::Shape& shape) {
    size_t input_size = ov::shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input.get_element_type()) {
    case ov::element::Type_t::i8: {
        auto p = input.data<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i16: {
        auto p = input.data<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i32: {
        auto p = input.data<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i64: {
        auto p = input.data<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u8: {
        auto p = input.data<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u16: {
        auto p = input.data<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u32: {
        auto p = input.data<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u64: {
        auto p = input.data<uint64_t>();
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

std::vector<int64_t> get_signal_size(const ov::TensorVector& inputs, size_t num_of_axes) {
    if (inputs.size() == 3) {
        return get_integers(inputs[2], inputs[2].get_shape());
    }

    return std::vector<int64_t>(num_of_axes, static_cast<int64_t>(-1));
}

ov::runtime::interpreter::EvaluatorsMap& ov::runtime::interpreter::get_evaluators_map() {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    OPENVINO_SUPPRESS_DEPRECATED_END
    return evaluatorsMap;
}
