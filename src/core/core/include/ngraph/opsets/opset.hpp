// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>

#include "ngraph/factory.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    /// \brief Run-time opset information
    class NGRAPH_API OpSet
    {
        static std::mutex& get_mutex();

    public:
        OpSet() {}
        std::set<NodeTypeInfo>::size_type size() const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.size();
        }
        /// \brief Insert an op into the opset with a particular name and factory
        void insert(const std::string& name,
                    const NodeTypeInfo& type_info,
                    FactoryRegistry<Node>::Factory factory)
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            m_op_types.insert(type_info);
            m_name_type_info_map[name] = type_info;
            m_case_insensitive_type_info_map[to_upper_name(name)] = type_info;
            m_factory_registry.register_factory(type_info, factory);
        }

        /// \brief Insert OP_TYPE into the opset with a special name and the default factory
        template <typename OP_TYPE>
        void insert(const std::string& name)
        {
            insert(name, OP_TYPE::type_info, FactoryRegistry<Node>::get_default_factory<OP_TYPE>());
        }

        /// \brief Insert OP_TYPE into the opset with the default name and factory
        template <typename OP_TYPE>
        void insert()
        {
            insert<OP_TYPE>(OP_TYPE::type_info.name);
        }

        const std::set<NodeTypeInfo>& get_types_info() const { return m_op_types; }
        /// \brief Create the op named name using it's factory
        ngraph::Node* create(const std::string& name) const;

        /// \brief Create the op named name using it's factory
        ngraph::Node* create_insensitive(const std::string& name) const;

        /// \brief Return true if OP_TYPE is in the opset
        bool contains_type(const NodeTypeInfo& type_info) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(type_info) != m_op_types.end();
        }

        /// \brief Return true if OP_TYPE is in the opset
        template <typename OP_TYPE>
        bool contains_type() const
        {
            return contains_type(OP_TYPE::type_info);
        }

        /// \brief Return true if name is in the opset
        bool contains_type(const std::string& name) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_name_type_info_map.find(name) != m_name_type_info_map.end();
        }

        /// \brief Return true if name is in the opset
        bool contains_type_insensitive(const std::string& name) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_case_insensitive_type_info_map.find(to_upper_name(name)) !=
                   m_case_insensitive_type_info_map.end();
        }

        /// \brief Return true if node's type is in the opset
        bool contains_op_type(const Node* node) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(node->get_type_info()) != m_op_types.end();
        }

        const std::set<NodeTypeInfo>& get_type_info_set() const { return m_op_types; }
        ngraph::FactoryRegistry<ngraph::Node>& get_factory_registry() { return m_factory_registry; }

    protected:
        static std::string to_upper_name(const std::string& name)
        {
            std::string upper_name = name;
            std::locale loc;
            std::transform(upper_name.begin(),
                           upper_name.end(),
                           upper_name.begin(),
                           [&loc](char c) { return std::toupper(c, loc); });
            return upper_name;
        }

        ngraph::FactoryRegistry<ngraph::Node> m_factory_registry;
        std::set<NodeTypeInfo> m_op_types;
        std::map<std::string, NodeTypeInfo> m_name_type_info_map;
        std::map<std::string, NodeTypeInfo> m_case_insensitive_type_info_map;
    };

    const NGRAPH_API OpSet& get_opset1();
    const NGRAPH_API OpSet& get_opset2();
    const NGRAPH_API OpSet& get_opset3();
    const NGRAPH_API OpSet& get_opset4();
    const NGRAPH_API OpSet& get_opset5();
    const NGRAPH_API OpSet& get_opset6();
    const NGRAPH_API OpSet& get_opset7();
    const NGRAPH_API OpSet& get_opset8();
} // namespace ngraph
