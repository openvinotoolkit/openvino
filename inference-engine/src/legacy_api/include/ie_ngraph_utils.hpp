// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_precision.hpp>
#include <ngraph/type/element_type.hpp>
#include <string>
#include <algorithm>

namespace InferenceEngine {
namespace details {

inline ::ngraph::element::Type convertPrecision(const Precision& precision) {
    Precision::ePrecision pType = precision;
    switch (pType) {
    case Precision::UNSPECIFIED:
        return ::ngraph::element::Type(::ngraph::element::Type_t::undefined);
    case Precision::FP32:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f32);
    case Precision::FP16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f16);
    case Precision::BF16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::bf16);
    case Precision::U8:
        return ::ngraph::element::Type(::ngraph::element::Type_t::u8);
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
    case Precision::Q78:
    case Precision::MIXED:
    case Precision::CUSTOM:
    default:
        THROW_IE_EXCEPTION << "Incorrect precision!";
    }
}

inline ::ngraph::element::Type convertPrecision(const std::string& precision) {
    if (precision == "f16" || precision == "FP16") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::f16);
    } else if (precision == "f32" || precision == "FP32") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::f32);
    } else if (precision == "bf16" || precision == "BF16") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::bf16);
    } else if (precision == "f64" || precision == "FP64") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::f64);
    } else if (precision == "i8" || precision == "I8") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::i8);
    } else if (precision == "i16" || precision == "I16") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::i16);
    } else if (precision == "i32" || precision == "I32") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::i32);
    } else if (precision == "i64" || precision == "I64") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::i64);
    } else if (precision == "u1" || precision == "U1") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::u1);
    } else if (precision == "u8" || precision == "U8") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::u8);
    } else if (precision == "u16" || precision == "U16") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::u16);
    } else if (precision == "u32" || precision == "U32") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::u32);
    } else if (precision == "u64" || precision == "U64") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::u64);
    } else if (precision == "boolean" || precision == "BOOL") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::boolean);
    } else if (precision == "undefined") {
        return ::ngraph::element::Type(::ngraph::element::Type_t::undefined);
    } else {
        THROW_IE_EXCEPTION << "Incorrect precision: " << precision;
    }
}

inline Precision convertPrecision(const ::ngraph::element::Type& precision) {
    switch (precision.get_type_enum()) {
    case ::ngraph::element::Type_t::undefined:
        return Precision(Precision::UNSPECIFIED);
    case ::ngraph::element::Type_t::f16:
        return Precision(Precision::FP16);
    case ::ngraph::element::Type_t::f32:
        return Precision(Precision::FP32);
    case ::ngraph::element::Type_t::bf16:
        return Precision(Precision::BF16);
    case ::ngraph::element::Type_t::i8:
        return Precision(Precision::I8);
    case ::ngraph::element::Type_t::i16:
        return Precision(Precision::I16);
    case ::ngraph::element::Type_t::i32:
        return Precision(Precision::I32);
    case ::ngraph::element::Type_t::i64:
        return Precision(Precision::I64);
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
    default:
        THROW_IE_EXCEPTION << "Incorrect precision " << precision.get_type_name() << "!";
    }
}

}  // namespace details
}  // namespace InferenceEngine
