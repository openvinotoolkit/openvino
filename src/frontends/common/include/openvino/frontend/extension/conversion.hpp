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
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {

template <class T>
class FRONTEND_API ConversionExtensionBase : public ov::Extension {
public:
    ConversionExtensionBase(const std::string& op_type, const CreatorFunction<T>& converter)
        : m_op_type(op_type),
          m_converter(converter) {}

    const CreatorFunction<T>& get_converter() const {
        return m_converter;
    }

    const std::string& get_op_type() const {
        return m_op_type;
    }

    ~ConversionExtensionBase() override = 0;
private:
    std::string m_op_type;
    CreatorFunction<T> m_converter;
};

template <class T>
class FRONTEND_API ConversionExtension : public ConversionExtensionBase<T> {
public:
    ConversionExtension(const std::string& op_type, const CreatorFunction<T>& converter)
        : ConversionExtensionBase<T>(op_type, converter) {}
};

}  // namespace frontend
}  // namespace ov
