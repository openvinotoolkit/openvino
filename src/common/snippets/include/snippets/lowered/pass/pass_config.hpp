// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/core/type.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface PassConfig
 * @brief Represents a transformations config that is used for disabling/enabling
 *        passes registered inside lowered::pass::PassPipeline
 * @ingroup snippets
 */
class PassConfig {
public:
    PassConfig() = default;

    void disable(const DiscreteTypeInfo& type_info);
    template <class T>
    void disable() {
        disable(T::get_type_info_static());
    }

    void enable(const DiscreteTypeInfo& type_info);
    template <class T>
    void enable() {
        enable(T::get_type_info_static());
    }

    bool is_disabled(const DiscreteTypeInfo& type_info) const;
    template <class T>
    bool is_disabled() const {
        return is_disabled(T::get_type_info_static());
    }

    bool is_enabled(const DiscreteTypeInfo& type_info) const;
    template <class T>
    bool is_enabled() const {
        return is_enabled(T::get_type_info_static());
    }

    friend bool operator==(const PassConfig& lhs, const PassConfig& rhs);
    friend bool operator!=(const PassConfig& lhs, const PassConfig& rhs);

private:
    std::unordered_set<DiscreteTypeInfo> m_disabled;
    std::unordered_set<DiscreteTypeInfo> m_enabled;
};

}  // namespace ov::snippets::lowered::pass
