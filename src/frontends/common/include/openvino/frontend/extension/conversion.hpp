// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {

class FRONTEND_API ConversionExtensionBase : public ov::Extension {
public:
    explicit ConversionExtensionBase(const std::string& op_type) : m_op_type(op_type) {}

    const std::string& get_op_type() const {
        return m_op_type;
    }

    ~ConversionExtensionBase() override = 0;

private:
    std::string m_op_type;
};

class FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    ConversionExtension(const std::string& op_type, const CreatorFunctionNamed& converter)
        : ConversionExtensionBase(op_type),
          m_converter_named(converter) {}

    const CreatorFunction& get_converter() {
        return m_converter;
    };

    const CreatorFunctionNamed& get_converter_named() {
        return m_converter_named;
    };

    ~ConversionExtension() override = default;

private:
    // todo: try to use ov::Any
    CreatorFunction m_converter;
    CreatorFunctionNamed m_converter_named;
};

}  // namespace frontend
}  // namespace ov
