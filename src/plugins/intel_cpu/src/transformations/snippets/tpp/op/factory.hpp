// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/core/type.hpp"
#include "openvino/pass/pattern/matcher.hpp"
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
    static std::shared_ptr<ov::Node> create(const std::shared_ptr<ov::Node>& n);
    static bool is_supported(const std::shared_ptr<ov::Node>& n);
    typedef std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> tpp_builder;
    typedef std::function<bool(const std::shared_ptr<ov::Node>&)> tpp_matcher;
    struct TPPCustomBuilder {
        tpp_matcher matcher;
        tpp_builder builder;
    };
private:
    static std::unordered_map<ov::DiscreteTypeInfo, tpp_builder> m_direct_mappig;
    static std::vector<TPPCustomBuilder> m_custom_mapping;
};
} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
