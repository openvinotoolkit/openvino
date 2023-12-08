// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>

#include "cpp/ie_cnn_network.h"
#include "ie_precision.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"

namespace InferenceEngine {
namespace details {

INFERENCE_ENGINE_1_0_DEPRECATED inline ::ngraph::element::Type convertPrecision(const Precision& precision) {
    Precision::ePrecision pType = precision;
    switch (pType) {
    case Precision::UNSPECIFIED:
        return ::ngraph::element::Type(::ngraph::element::Type_t::undefined);
    case Precision::FP32:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f32);
    case Precision::FP64:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f64);
    case Precision::FP16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f16);
    case Precision::BF16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::bf16);
    case Precision::U4:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u4);
    case Precision::U8:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u8);
    case Precision::I4:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i4);
    case Precision::I8:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i8);
    case Precision::U16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u16);
    case Precision::I16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i16);
    case Precision::I32:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i32);
    case Precision::U32:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u32);
    case Precision::I64:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i64);
    case Precision::U64:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u64);
    case Precision::BOOL:
        return ::ngraph::element::Type(::ngraph::element::Type_t::boolean);
    case Precision::BIN:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u1);
    case Precision::NF4:
        return ::ngraph::element::Type(::ngraph::element::Type_t::nf4);
    case Precision::STRING:
        return ::ngraph::element::Type(::ngraph::element::Type_t::string);
    case Precision::Q78:
    case Precision::MIXED:
    case Precision::CUSTOM:
    default:
        IE_THROW() << "Incorrect precision!";
    }
}

INFERENCE_ENGINE_1_0_DEPRECATED inline ::ngraph::element::Type convertPrecision(const std::string& precision) {
    return ::ov::element::Type(precision);
}

INFERENCE_ENGINE_1_0_DEPRECATED inline Precision convertPrecision(const ::ngraph::element::Type& precision) {
    switch (precision) {
    case ::ngraph::element::Type_t::undefined:
        return Precision(Precision::UNSPECIFIED);
    case ::ngraph::element::Type_t::f16:
        return Precision(Precision::FP16);
    case ::ngraph::element::Type_t::f32:
        return Precision(Precision::FP32);
    case ::ngraph::element::Type_t::f64:
        return Precision(Precision::FP64);
    case ::ngraph::element::Type_t::bf16:
        return Precision(Precision::BF16);
    case ::ngraph::element::Type_t::i4:
        return Precision(Precision::I4);
    case ::ngraph::element::Type_t::i8:
        return Precision(Precision::I8);
    case ::ngraph::element::Type_t::i16:
        return Precision(Precision::I16);
    case ::ngraph::element::Type_t::i32:
        return Precision(Precision::I32);
    case ::ngraph::element::Type_t::i64:
        return Precision(Precision::I64);
    case ::ngraph::element::Type_t::u4:
        return Precision(Precision::U4);
    case ::ngraph::element::Type_t::u8:
        return Precision(Precision::U8);
    case ::ngraph::element::Type_t::u16:
        return Precision(Precision::U16);
    case ::ngraph::element::Type_t::u32:
        return Precision(Precision::U32);
    case ::ngraph::element::Type_t::u64:
        return Precision(Precision::U64);
    case ::ngraph::element::Type_t::u1:
        return Precision(Precision::BIN);
    case ::ngraph::element::Type_t::boolean:
        return Precision(Precision::BOOL);
    case ::ngraph::element::Type_t::nf4:
        return Precision(Precision::NF4);
    case ::ngraph::element::Type_t::string:
        return Precision(Precision::STRING);
    case ::ngraph::element::Type_t::dynamic:
        return Precision(Precision::UNSPECIFIED);
    default:
        IE_THROW() << "Incorrect precision " << precision.to_string() << "!";
        return {};
    }
}

/**
 * @brief Clones input network including all layers and internal data objects
 * @note Blobs inside layers are reused
 * @param network A network to clone
 * @return A cloned object
 */
INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CPP(CNNNetwork) cloneNetwork(const CNNNetwork& network);

}  // namespace details
}  // namespace InferenceEngine
