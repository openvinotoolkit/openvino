// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ie_precision.hpp"
#include "openvino/core/type/element_type.hpp"

namespace FuncTestUtils {
namespace PrecisionUtils {

// Copied from inference-engine/src/inference_engine/src/ie_ngraph_utils.hpp
inline ::ov::element::Type convertIE2nGraphPrc(const InferenceEngine::Precision& precision) {
    InferenceEngine::Precision::ePrecision pType = precision;
    switch (pType) {
    case InferenceEngine::Precision::UNSPECIFIED:
        return ::ov::element::Type(::ov::element::Type_t::undefined);
    case InferenceEngine::Precision::FP64:
        return ::ov::element::Type(::ov::element::Type_t::f64);
    case InferenceEngine::Precision::FP32:
        return ::ov::element::Type(::ov::element::Type_t::f32);
    case InferenceEngine::Precision::FP16:
        return ::ov::element::Type(::ov::element::Type_t::f16);
    case InferenceEngine::Precision::BF16:
        return ::ov::element::Type(::ov::element::Type_t::bf16);
    case InferenceEngine::Precision::U4:
        return ::ov::element::Type(::ov::element::Type_t::u4);
    case InferenceEngine::Precision::I4:
        return ::ov::element::Type(::ov::element::Type_t::i4);
    case InferenceEngine::Precision::U8:
        return ::ov::element::Type(::ov::element::Type_t::u8);
    case InferenceEngine::Precision::I8:
        return ::ov::element::Type(::ov::element::Type_t::i8);
    case InferenceEngine::Precision::U16:
        return ::ov::element::Type(::ov::element::Type_t::u16);
    case InferenceEngine::Precision::I16:
        return ::ov::element::Type(::ov::element::Type_t::i16);
    case InferenceEngine::Precision::U32:
        return ::ov::element::Type(::ov::element::Type_t::u32);
    case InferenceEngine::Precision::I32:
        return ::ov::element::Type(::ov::element::Type_t::i32);
    case InferenceEngine::Precision::I64:
        return ::ov::element::Type(::ov::element::Type_t::i64);
    case InferenceEngine::Precision::U64:
        return ::ov::element::Type(::ov::element::Type_t::u64);
    case InferenceEngine::Precision::BOOL:
        return ::ov::element::Type(::ov::element::Type_t::boolean);
    case InferenceEngine::Precision::BIN:
        return ::ov::element::Type(::ov::element::Type_t::u1);
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::MIXED:
    case InferenceEngine::Precision::CUSTOM:
    default:
        IE_THROW() << "Incorrect precision!";
    }
}

}  // namespace PrecisionUtils
}  // namespace FuncTestUtils
