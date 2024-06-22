// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/operator_set.hpp"
#include "version_range.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class OperatorsBridge {
public:
    OperatorsBridge();

    OperatorsBridge(const OperatorsBridge&) = default;
    OperatorsBridge& operator=(const OperatorsBridge&) = default;
    OperatorsBridge(OperatorsBridge&&) = default;
    OperatorsBridge& operator=(OperatorsBridge&&) = default;

    OperatorSet get_operator_set(const std::string& domain, std::int64_t version = -1) const;

    template <typename Container = std::set<std::string>>
    Container get_supported_operators(int64_t version, std::string domain) const {
        if (domain == "ai.onnx") {
            domain = "";
        }

        Container ops{};
        const auto dm = m_map.find(domain);
        if (dm == std::end(m_map)) {
            return ops;
        }

        std::insert_iterator<Container> inserter{ops, std::begin(ops)};

        std::transform(std::begin(dm->second),
                       std::end(dm->second),
                       inserter,
                       [](const DomainOpset::value_type& op_in_domain) {
                           return op_in_domain.first;
                       });

        return ops;
    }

    void register_operator(const std::string& name, std::int64_t version, const std::string& domain, Operator fn);
    void unregister_operator(const std::string& name, std::int64_t version, const std::string& domain);
    bool is_operator_registered(const std::string& name, std::int64_t version, const std::string& domain) const;

    void overwrite_operator(const std::string& name, const std::string& domain, Operator fn);

private:
    void register_operator_in_custom_domain(std::string name,
                                            ov::frontend::onnx::VersionRange range,
                                            Operator fn,
                                            std::string domain,
                                            std::string warning_mes = "");
    void register_operator(std::string name,
                           ov::frontend::onnx::VersionRange range,
                           Operator fn,
                           std::string warning_mes = "");
    // Registered operators structure
    // {
    //    domain_1: {
    //      op_type_1: {
    //          version_1: {func_handle},
    //          version_2: {func_handle},
    //          ...
    //      },
    //      op_type_2: { ... }
    //      ...
    //    },
    //    domain_2: { ... },
    //    ...
    // }
    std::unordered_map<std::string, DomainOpset> m_map;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
