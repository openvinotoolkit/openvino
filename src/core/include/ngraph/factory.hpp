// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <functional>
#include <mutex>
#include <unordered_map>

#include "ngraph/deprecated.hpp"
#include "ngraph/ngraph_visibility.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
NGRAPH_API_DEPRECATED NGRAPH_API std::mutex& get_registry_mutex();

/// \brief Registry of factories that can construct objects derived from BASE_TYPE
template <typename BASE_TYPE>
class FactoryRegistry {
public:
    using Factory = std::function<BASE_TYPE*()>;
    using FactoryMap = std::unordered_map<typename BASE_TYPE::type_info_t, Factory>;

    // \brief Get the default factory for DERIVED_TYPE. Specialize as needed.
    template <typename DERIVED_TYPE>
    static Factory get_default_factory() {
        return []() {
            return new DERIVED_TYPE();
        };
    }

    ~FactoryRegistry() = default;

    /// \brief Register a custom factory for type_info
    void register_factory(const typename BASE_TYPE::type_info_t& type_info, Factory factory) {
        std::lock_guard<std::mutex> guard(get_registry_mutex());
        m_factory_map[type_info] = factory;
    }

    /// \brief Register a custom factory for DERIVED_TYPE
    template <typename DERIVED_TYPE>
    void register_factory(Factory factory) {
        register_factory(DERIVED_TYPE::get_type_info_static(), factory);
    }

    /// \brief Register the defualt constructor factory for DERIVED_TYPE
    template <typename DERIVED_TYPE>
    void register_factory() {
        register_factory<DERIVED_TYPE>(get_default_factory<DERIVED_TYPE>());
    }

    /// \brief Check to see if a factory is registered
    bool has_factory(const typename BASE_TYPE::type_info_t& info) {
        std::lock_guard<std::mutex> guard(get_registry_mutex());
        return m_factory_map.find(info) != m_factory_map.end();
    }

    /// \brief Check to see if DERIVED_TYPE has a registered factory
    template <typename DERIVED_TYPE>
    bool has_factory() {
        return has_factory(DERIVED_TYPE::get_type_info_static());
    }

    /// \brief Create an instance for type_info
    BASE_TYPE* create(const typename BASE_TYPE::type_info_t& type_info) const {
        std::lock_guard<std::mutex> guard(get_registry_mutex());
        auto it = m_factory_map.find(type_info);
        return it == m_factory_map.end() ? nullptr : it->second();
    }

    /// \brief Create an instance using factory for DERIVED_TYPE
    template <typename DERIVED_TYPE>
    BASE_TYPE* create() const {
        return create(DERIVED_TYPE::get_type_info_static());
    }

protected:
    FactoryMap m_factory_map;
};
}  // namespace ngraph
NGRAPH_SUPPRESS_DEPRECATED_END
