// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/extension/conversion.hpp"

using namespace ov::frontend::tensorflow;

ConversionExtension::ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
    : ConversionExtensionBase(op_type),
      m_converter(converter) {}

const ov::frontend::CreatorFunction& ConversionExtension::get_converter() const {
    return m_converter;
}

ConversionExtension ::~ConversionExtension() = default;
