// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ops_bridge.hpp"

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "core/attribute.hpp"
#include "openvino/util/log.hpp"
#if !defined(ONNX_BUILD_SHARED)
#    include "static_reg.hpp"
#endif

using namespace ov::frontend::onnx;

namespace ov {
namespace frontend {
namespace onnx {

namespace {
template <typename Container = std::map<int64_t, Operator>>
typename Container::const_iterator find(int64_t version, const Container& map) {
    // Get the latest version.
    if (version == -1) {
        return map.empty() ? std::end(map) : --std::end(map);
    }
    while (version > 0) {
        const auto it = map.find(version--);
        if (it != std::end(map)) {
            return it;
        }
    }
    return std::end(map);
}
}  // namespace

// Known domains (see operator_set.hpp for a declaration)
const char* OPENVINO_ONNX_DOMAIN = "org.openvinotoolkit";
const char* MICROSOFT_DOMAIN = "com.microsoft";
const char* PYTORCH_ATEN_DOMAIN = "org.pytorch.aten";
const char* MMDEPLOY_DOMAIN = "mmdeploy";

// Central storage of supported translators for operations
typedef std::unordered_map<std::string, DomainOpset> SupportedOps;
SupportedOps* get_supported_ops(void) {
    static SupportedOps supported_ops;
    return &supported_ops;
}

bool register_translator_exact(const std::string& name,
                               const int64_t exact_version,
                               const Operator fn,
                               const std::string& domain) {
    auto& supported_ops = *get_supported_ops();
    auto it = supported_ops[domain][name].find(exact_version);
    if (it == supported_ops[domain][name].end()) {
        supported_ops[domain][name].emplace(exact_version, std::bind(fn, std::placeholders::_1));
        return true;
    } else {
        // Left this option to be able create some custom operators which overwrites existing
        it->second = std::move(fn);
    }
    return false;
}

bool register_translator(const std::string name,
                         const VersionRange range,
                         const Operator fn,
                         const std::string domain) {
    for (int version = range.m_since; version <= range.m_until; ++version) {
        register_translator_exact(name, version, fn, domain);
    }
    return true;
}

void OperatorsBridge::register_operator_in_custom_domain(std::string name,
                                                         VersionRange range,
                                                         Operator fn,
                                                         std::string domain,
                                                         std::string warning_mes) {
    for (int version = range.m_since; version <= range.m_until; ++version) {
        register_operator(name, version, domain, fn);
    }
    if (!warning_mes.empty()) {
        OPENVINO_WARN("Operator: ",
                      name,
                      " since version: ",
                      range.m_since,
                      " until version: ",
                      range.m_until,
                      " registered with warning: ",
                      warning_mes);
    }
}

void OperatorsBridge::register_operator(std::string name, VersionRange range, Operator fn, std::string warning_mes) {
    register_operator_in_custom_domain(name, range, std::move(fn), "", warning_mes);
}

void OperatorsBridge::register_operator(const std::string& name,
                                        int64_t version,
                                        const std::string& domain,
                                        Operator fn) {
    auto it = m_map[domain][name].find(version);
    if (it == std::end(m_map[domain][name])) {
        m_map[domain][name].emplace(version, std::move(fn));
    } else {
        it->second = std::move(fn);
        OPENVINO_WARN("Overwriting existing operator: ",
                      (domain.empty() ? "ai.onnx" : domain),
                      ".",
                      name,
                      ":",
                      std::to_string(version));
    }
}

void OperatorsBridge::unregister_operator(const std::string& name, int64_t version, const std::string& domain) {
    auto domain_it = m_map.find(domain);
    if (domain_it == m_map.end()) {
        OPENVINO_ERR("unregister_operator: domain '", domain, "' was not registered before");
        return;
    }
    auto name_it = domain_it->second.find(name);
    if (name_it == domain_it->second.end()) {
        OPENVINO_ERR("unregister_operator: operator '", name, "' was not registered before");
        return;
    }
    auto version_it = name_it->second.find(version);
    if (version_it == name_it->second.end()) {
        OPENVINO_ERR("unregister_operator: operator '",
                     name,
                     "' with version ",
                     std::to_string(version),
                     " was not registered before");
        return;
    }
    m_map[domain][name].erase(version_it);
    if (m_map[domain][name].empty()) {
        m_map[domain].erase(name);
        if (m_map[domain].empty()) {
            m_map.erase(domain);
        }
    }
}

OperatorSet OperatorsBridge::get_operator_set(const std::string& domain, int64_t version) const {
    OperatorSet result;

    const auto dm = m_map.find(domain);
    if (dm == std::end(m_map)) {
        OPENVINO_DEBUG("Domain not recognized by OpenVINO");
        return result;
    }
    if (domain == "" && version > LATEST_SUPPORTED_ONNX_OPSET_VERSION) {
        OPENVINO_WARN("Currently ONNX operator set version: ",
                      version,
                      " is unsupported. Falling back to: ",
                      LATEST_SUPPORTED_ONNX_OPSET_VERSION);
    }
    for (const auto& op : dm->second) {
        const auto& it = find(version, op.second);
        if (it == std::end(op.second)) {
            OPENVINO_THROW("Unsupported operator version: " + (domain.empty() ? "" : domain + ".") + op.first + ":" +
                           std::to_string(version));
        }
        result.emplace(op.first, it->second);
    }
    return result;
}

bool OperatorsBridge::is_operator_registered(const std::string& name,
                                             int64_t version,
                                             const std::string& domain) const {
    // search for domain
    const auto dm_map = m_map.find(domain);
    if (dm_map == std::end(m_map)) {
        return false;
    }
    // search for name
    const auto op_map = dm_map->second.find(name);
    if (op_map == std::end(dm_map->second)) {
        return false;
    }

    return find(version, op_map->second) != std::end(op_map->second);
}

void OperatorsBridge::overwrite_operator(const std::string& name, const std::string& domain, Operator fn) {
    const auto domain_it = m_map.find(domain);
    if (domain_it != m_map.end()) {
        auto& domain_opset = domain_it->second;
        domain_opset[name].clear();
    }
    register_operator(name, 1, domain, std::move(fn));
}

#define REGISTER_OPERATOR(name_, ver_, fn_) \
    m_map[""][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1));

#define REGISTER_OPERATOR_WITH_DOMAIN(domain_, name_, ver_, fn_) \
    m_map[domain_][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1));

OperatorsBridge::OperatorsBridge() {
    // Deep copy of default map to local
    for (auto& domain : *get_supported_ops()) {
        for (auto& operation : domain.second) {
            for (auto& version : operation.second) {
                m_map[domain.first][operation.first].emplace(version.first, version.second);
            }
        }
    }
    // custom ops
}

#undef REGISTER_OPERATOR
#undef REGISTER_OPERATOR_WITH_DOMAIN
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
