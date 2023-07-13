// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <mutex>

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"
#include "ops_bridge.hpp"

namespace ngraph {
namespace onnx_import {
/// An extension holding its own copy of the OperatorsBridge which should only be used with legacy ONNX importer API
/// Having it here keeps the legacy API operational without interfering with the frontends API
class LegacyConversionExtension : public ov::frontend::ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<LegacyConversionExtension>;

    LegacyConversionExtension() : ov::frontend::ConversionExtensionBase("") {}

    OperatorsBridge& ops_bridge() {
        return m_legacy_ops_bridge;
    }

    /// The legacy API entry point for registering custom operations globally (does not affect ONNX FE)
    void register_operator(const std::string& name, int64_t version, const std::string& domain, Operator fn) {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_legacy_ops_bridge.register_operator(name, version, domain, std::move(fn));
    }

    void unregister_operator(const std::string& name, int64_t version, const std::string& domain) {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_legacy_ops_bridge.unregister_operator(name, version, domain);
    }

private:
    std::mutex m_mutex;
    OperatorsBridge m_legacy_ops_bridge;
};
}  // namespace onnx_import
}  // namespace ngraph
