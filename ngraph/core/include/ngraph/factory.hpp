//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <functional>
#include <mutex>
#include <unordered_map>

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    NGRAPH_API std::mutex& get_registry_mutex();

    /// \brief Registry of factories that can construct objects derived from BASE_TYPE
    template <typename BASE_TYPE>
    class FactoryRegistry
    {
    public:
        using Factory = std::function<BASE_TYPE*()>;
        using FactoryMap = std::unordered_map<typename BASE_TYPE::type_info_t, Factory>;

        // \brief Get the default factory for DERIVED_TYPE. Specialize as needed.
        template <typename DERIVED_TYPE>
        static Factory get_default_factory()
        {
            return []() { return new DERIVED_TYPE(); };
        }

        /// \brief Register a custom factory for type_info
        void register_factory(const typename BASE_TYPE::type_info_t& type_info, Factory factory)
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            m_factory_map[type_info] = factory;
        }

        /// \brief Register a custom factory for DERIVED_TYPE
        template <typename DERIVED_TYPE>
        void register_factory(Factory factory)
        {
            register_factory(DERIVED_TYPE::type_info, factory);
        }

        /// \brief Register the defualt constructor factory for DERIVED_TYPE
        template <typename DERIVED_TYPE>
        void register_factory()
        {
            register_factory<DERIVED_TYPE>(get_default_factory<DERIVED_TYPE>());
        }

        /// \brief Check to see if a factory is registered
        bool has_factory(const typename BASE_TYPE::type_info_t& info)
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            return m_factory_map.find(info) != m_factory_map.end();
        }

        /// \brief Check to see if DERIVED_TYPE has a registered factory
        template <typename DERIVED_TYPE>
        bool has_factory()
        {
            return has_factory(DERIVED_TYPE::type_info);
        }

        /// \brief Create an instance for type_info
        BASE_TYPE* create(const typename BASE_TYPE::type_info_t& type_info) const
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            auto it = m_factory_map.find(type_info);
            return it == m_factory_map.end() ? nullptr : it->second();
        }

        /// \brief Create an instance using factory for DERIVED_TYPE
        template <typename DERIVED_TYPE>
        BASE_TYPE* create() const
        {
            return create(DERIVED_TYPE::type_info);
        }

        /// \brief Get the factory for BASE_TYPE
        static FactoryRegistry<BASE_TYPE>& get();

    protected:
        FactoryMap m_factory_map;
    };
} // namespace ngraph
