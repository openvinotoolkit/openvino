// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_precision.hpp>
#include <ngraph/type/element_type.hpp>

namespace InferenceEngine {
namespace details {
namespace ngraph {

inline ::ngraph::element::Type convertPrecision(const Precision& precision) {
    Precision::ePrecision pType = precision;
    switch (pType) {
    case Precision::UNSPECIFIED:
        return ::ngraph::element::Type(::ngraph::element::Type_t::undefined);
    case Precision::FP32:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f32);
    case Precision::FP16:
        return ::ngraph::element::Type(::ngraph::element::Type_t::f16);
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
    case Precision::I64:
        return ::ngraph::element::Type(::ngraph::element::Type_t::i64);
    case Precision::BIN:
        return ::ngraph::element::Type(::ngraph::element::Type_t::boolean);
    case Precision::Q78:
    case Precision::MIXED:
    case Precision::CUSTOM:
    default:
        THROW_IE_EXCEPTION << "Incorrect precision!";
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
    default:
        THROW_IE_EXCEPTION << "Incorrect precision!";
    }
}

}  // namespace ngraph
}  // namespace details
}  // namespace InferenceEngine


