// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/conversion.hpp"

using namespace ov::frontend;
ConversionExtensionBase::~ConversionExtensionBase() = default;

ConversionExtension::ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
    : ConversionExtensionBase(op_type),
      m_converter(converter) {}

ConversionExtension::ConversionExtension(const std::string& op_type, const CreatorFunctionNamed& converter)
    : ConversionExtensionBase(op_type),
      m_converter_named(converter) {}

const CreatorFunction& ConversionExtension::get_converter() const {
    return m_converter;
};

const CreatorFunctionNamed& ConversionExtension::get_converter_named() const {
    return m_converter_named;
};

ConversionExtension::~ConversionExtension() = default;
