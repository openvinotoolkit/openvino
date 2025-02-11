// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu::tpp::op {
class NodeFactory {
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
    static std::unordered_map<ov::DiscreteTypeInfo, tpp_builder> m_direct_mapping;
    static std::vector<TPPCustomBuilder> m_custom_mapping;
};
}  // namespace ov::intel_cpu::tpp::op
