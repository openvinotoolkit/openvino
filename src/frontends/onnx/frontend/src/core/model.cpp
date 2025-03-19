// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/model.hpp"

#include <onnx/onnx_pb.h>

#include "onnx_framework_node.hpp"
#include "openvino/util/log.hpp"
#include "ops_bridge.hpp"

using namespace ::ONNX_NAMESPACE;

namespace ov {
namespace frontend {
namespace onnx {
std::string get_node_domain(const NodeProto& node_proto) {
    return node_proto.has_domain() ? node_proto.domain() : "";
}

std::int64_t get_opset_version(const ModelProto& model_proto, const std::string& domain) {
    // copy the opsets and sort them (descending order)
    // then return the version from the first occurrence of a given domain
    auto opset_imports = model_proto.opset_import();
    std::sort(std::begin(opset_imports),
              std::end(opset_imports),
              [](const OperatorSetIdProto& lhs, const OperatorSetIdProto& rhs) {
                  return lhs.version() > rhs.version();
              });

    for (const auto& opset_import : opset_imports) {
        if (domain == opset_import.domain()) {
            return opset_import.version();
        }
    }

    OPENVINO_THROW("Couldn't find operator set's version for domain: ", domain, ".");
}

Model::Model(std::shared_ptr<ModelProto> model_proto, ModelOpSet&& model_opset)
    : m_model_proto{std::move(model_proto)},
      m_opset{std::move(model_opset)} {}

const Operator& Model::get_operator(const std::string& name, const std::string& domain) const {
    const auto dm = m_opset.find(domain);
    if (dm == std::end(m_opset)) {
        OPENVINO_THROW("Domain isn't supported: " + domain);
    }
    const auto op = dm->second.find(name);
    if (op == std::end(dm->second)) {
        OPENVINO_THROW("Operation isn't supported: " + (domain.empty() ? "" : domain + ".") + name);
    }
    return op->second;
}

bool Model::is_operator_available(const std::string& name, const std::string& domain) const {
    const auto dm = m_opset.find(domain);
    if (dm == std::end(m_opset)) {
        return false;
    }
    const auto op = dm->second.find(name);
    return (op != std::end(dm->second));
}

void Model::enable_opset_domain(const std::string& domain, const OperatorsBridge& ops_bridge) {
    // There is no need to 'update' already enabled domain.
    // Since this function may be called only during model import,
    // (maybe multiple times) the registered domain opset won't differ
    // between subsequent calls.
    if (m_opset.find(domain) == std::end(m_opset)) {
        const auto opset = ops_bridge.get_operator_set(domain);
        if (opset.empty()) {
            OPENVINO_WARN("Couldn't enable domain: ", domain, " since it does not have any registered operators.");
            return;
        }
        m_opset.emplace(domain, opset);
    }
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
