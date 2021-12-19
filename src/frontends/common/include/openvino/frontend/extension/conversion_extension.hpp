// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/any.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "tensorflow_frontend/node_context.hpp"

namespace ov {
namespace frontend {

using CreatorFunction = std::function<OutputVector(const NodeContext&)>;
using CreatorFunctionNamed = std::function<std::map<std::string, OutputVector>(const NodeContext&)>;
class FRONTEND_API ConversionExtensionBase : public ov::Extension {
public:
    ConversionExtensionBase(const std::string& op_type, const CreatorFunction& converter)
        : m_op_type(op_type),
          m_converter(converter) {}

    ConversionExtensionBase(const std::string& op_type, const CreatorFunctionNamed& converter)
        : m_op_type(op_type),
          m_named_converter(converter) {}

    const CreatorFunction& get_converter() const {
        return m_converter;
    }

    const CreatorFunctionNamed& get_named_converter() const {
        return m_named_converter;
    }

    const std::string& get_op_type() const {
        return m_op_type;
    }

private:
    std::string m_op_type;
    CreatorFunction m_converter;
    CreatorFunctionNamed m_named_converter;
};

class FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
        : ConversionExtensionBase(op_type, converter) {}

    ConversionExtension(const std::string& op_type, const CreatorFunctionNamed& converter)
        : ConversionExtensionBase(op_type, converter) {}
};

}  // namespace frontend
}  // namespace ov
