// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"
#include "ops_bridge.hpp"

namespace ngraph {
namespace onnx_import {
class LegacyConversionExtension : public ov::frontend::ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<LegacyConversionExtension>;

    LegacyConversionExtension() : ov::frontend::ConversionExtensionBase("") {}

    const OperatorsBridge& ops_bridge() const {
        return m_legacy_ops_bridge;
    }

    void register_operator(const std::string& name, int64_t version, const std::string& domain, Operator fn) {
        m_legacy_ops_bridge.register_operator(name, version, domain, std::move(fn));
    }

    void unregister_operator(const std::string& name, int64_t version, const std::string& domain) {
        m_legacy_ops_bridge.unregister_operator(name, version, domain);
    }

private:
    OperatorsBridge m_legacy_ops_bridge;
};
}  // namespace onnx_import
}  // namespace ngraph
