// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"

#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

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

private:
    std::unordered_set<DiscreteTypeInfo> m_disabled;
    std::unordered_set<DiscreteTypeInfo> m_enabled;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
