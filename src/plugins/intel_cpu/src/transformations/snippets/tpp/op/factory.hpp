// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/core/type.hpp"
// template <>
// struct std::hash<const ov::DiscreteTypeInfo> {
//     std::size_t operator()(const ov::DiscreteTypeInfo& key) const {
//         return key.hash();
//     }
// };

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
class TPPNodeFactory {
public:
    static std::shared_ptr<ov::Node> create(const std::shared_ptr<ov::Node>& n) {
        const auto& found = m_supported.find(n->get_type_info());
        if (found != m_supported.end()) {
            const auto& tpp_node = (found->second)(n);
            tpp_node->set_friendly_name(n->get_friendly_name());
            return tpp_node;
        }
        return nullptr;
    }
    static bool is_supported(const std::shared_ptr<ov::Node>& n) {
        return m_supported.count(n->get_type_info());
    }

private:
    typedef std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> tpp_creator;
    static std::unordered_map<ov::DiscreteTypeInfo, tpp_creator> m_supported;
};
} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
