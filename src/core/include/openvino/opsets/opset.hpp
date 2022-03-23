// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "ngraph/compatibility.hpp"
#include "ngraph/factory.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/node.hpp"

namespace ov {
/**
 * @brief Run-time opset information
 * @ingroup ov_opset_cpp_api
 */
class OPENVINO_API OpSet {
    static std::mutex& get_mutex();

public:
    OpSet() = default;
    virtual ~OpSet() = default;
    std::set<NodeTypeInfo>::size_type size() const {
        std::lock_guard<std::mutex> guard(get_mutex());
        return m_op_types.size();
    }

    /// \brief Insert OP_TYPE into the opset with a special name and the default factory
    template <typename OP_TYPE, typename std::enable_if<!ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert(const std::string& name) {
        insert(name, OP_TYPE::get_type_info_static(), ngraph::FactoryRegistry<Node>::get_default_factory<OP_TYPE>());
    }
    template <typename OP_TYPE, typename std::enable_if<ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert(const std::string& name) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        insert(name, OP_TYPE::type_info, ngraph::FactoryRegistry<Node>::get_default_factory<OP_TYPE>());
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    /// \brief Insert OP_TYPE into the opset with the default name and factory
    template <typename OP_TYPE, typename std::enable_if<!ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert() {
        insert<OP_TYPE>(OP_TYPE::get_type_info_static().name);
    }
    template <typename OP_TYPE, typename std::enable_if<ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert() {
        OPENVINO_SUPPRESS_DEPRECATED_START
        insert<OP_TYPE>(OP_TYPE::type_info.name);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    const std::set<NodeTypeInfo>& get_types_info() const {
        return m_op_types;
    }
    /// \brief Create the op named name using it's factory
    ov::Node* create(const std::string& name) const;

    /// \brief Create the op named name using it's factory
    ov::Node* create_insensitive(const std::string& name) const;

    /// \brief Return true if OP_TYPE is in the opset
    bool contains_type(const NodeTypeInfo& type_info) const {
        std::lock_guard<std::mutex> guard(get_mutex());
        return m_op_types.find(type_info) != m_op_types.end();
    }

    /// \brief Return true if OP_TYPE is in the opset
    template <typename OP_TYPE, typename std::enable_if<!ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    bool contains_type() const {
        return contains_type(OP_TYPE::get_type_info_static());
    }
    template <typename OP_TYPE, typename std::enable_if<ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    bool contains_type() const {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return contains_type(OP_TYPE::type_info);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    /// \brief Return true if name is in the opset
    bool contains_type(const std::string& name) const {
        std::lock_guard<std::mutex> guard(get_mutex());
        return m_name_type_info_map.find(name) != m_name_type_info_map.end();
    }

    /// \brief Return true if name is in the opset
    bool contains_type_insensitive(const std::string& name) const {
        std::lock_guard<std::mutex> guard(get_mutex());
        return m_case_insensitive_type_info_map.find(to_upper_name(name)) != m_case_insensitive_type_info_map.end();
    }

    /// \brief Return true if node's type is in the opset
    bool contains_op_type(const Node* node) const {
        std::lock_guard<std::mutex> guard(get_mutex());
        return m_op_types.find(node->get_type_info()) != m_op_types.end();
    }

    const std::set<NodeTypeInfo>& get_type_info_set() const {
        return m_op_types;
    }

protected:
    static std::string to_upper_name(const std::string& name) {
        std::string upper_name = name;
        std::locale loc;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), [&loc](char c) {
            return std::toupper(c, loc);
        });
        return upper_name;
    }

    ngraph::FactoryRegistry<ov::Node> m_factory_registry;
    std::set<NodeTypeInfo> m_op_types;
    std::map<std::string, NodeTypeInfo> m_name_type_info_map;
    std::map<std::string, NodeTypeInfo> m_case_insensitive_type_info_map;

    /// \brief Insert an op into the opset with a particular name and factory
    void insert(const std::string& name,
                const NodeTypeInfo& type_info,
                ngraph::FactoryRegistry<Node>::Factory factory) {
        std::lock_guard<std::mutex> guard(get_mutex());
        m_op_types.insert(type_info);
        m_name_type_info_map[name] = type_info;
        m_case_insensitive_type_info_map[to_upper_name(name)] = type_info;
        m_factory_registry.register_factory(type_info, std::move(factory));
    }
};

/**
 * @brief Returns opset1
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset1();
/**
 * @brief Returns opset2
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset2();
/**
 * @brief Returns opset3
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset3();
/**
 * @brief Returns opset4
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset4();
/**
 * @brief Returns opset5
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset5();
/**
 * @brief Returns opset6
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset6();
/**
 * @brief Returns opset7
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset7();
/**
 * @brief Returns opset8
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset8();
const OPENVINO_API OpSet& get_opset9();
}  // namespace ov
