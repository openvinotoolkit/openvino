// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <unordered_map>

#include "openvino/core/visibility.hpp"

namespace ov {

/**
 * @brief Registry of factories that can construct objects derived from BASE_TYPE.
 */
template <typename BASE_TYPE>
class FactoryRegistry {
public:
    /** @brief Factory function which create object using default ctor/ */
    using Factory = std::function<BASE_TYPE*()>;
    /** @brief Factory map hold object type_info as key and Factory function. */
    using FactoryMap = std::unordered_map<typename BASE_TYPE::type_info_t, Factory>;

    ~FactoryRegistry() = default;

    /** @brief Register the default constructor factory for DERIVED_TYPE. */
    template <typename DERIVED_TYPE>
    void register_factory() {
        m_factory_map[DERIVED_TYPE::get_type_info_static()] = get_default_factory<DERIVED_TYPE>();
    }

    /**
     * @brief Create an instance for type_info.
     *
     * @param type_info   The type_info used to find Factory in registry.
     * @return BASE_TYPE* Pointer to created object.
     */
    BASE_TYPE* create(const typename BASE_TYPE::type_info_t& type_info) const {
        auto it = m_factory_map.find(type_info);
        return it == m_factory_map.end() ? nullptr : it->second();
    }

    /**
     * @brief Create an instance using factory for DERIVED_TYPE.
     *
     * @tparam DERIVED_TYPE Derived class type from BASE_TYPE.
     * @return BASE_TYPE* Pointer to created object.
     */
    template <typename DERIVED_TYPE>
    BASE_TYPE* create() const {
        return create(DERIVED_TYPE::get_type_info_static());
    }

protected:
    FactoryMap m_factory_map;

private:
    /** @brief Get the default factory for DERIVED_TYPE. Specialize as needed. */
    template <typename DERIVED_TYPE>
    static Factory get_default_factory() {
        return []() {
            return new DERIVED_TYPE();
        };
    }
};
}  // namespace ov
