// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/node.hpp"

namespace ngraph {
namespace onnx_import {
/// \brief      GraphCache stores and provides access to ONNX graph initializers.
class GraphCache {
public:
    /// \brief      Add node to the cache or override the existing one.
    ///
    /// \note       GraphCache takes ownership of the node.
    ///
    /// \param[in]  name       The name of node added to the cache.
    /// \param[in]  node       The node added to the cache.
    void emplace_node(const std::string& name, Output<ov::Node>&& node);

    /// \brief      Remove node from the cache
    ///
    /// \param[in]  name       The name of node to be removed
    void remove_node(const std::string& name);

    /// \brief      Get the node from the cache
    ///
    /// \note       If the node is not found the ov::Exception is thrown.
    ///
    /// \param[in]  name       The name of the node.
    ///
    /// \return     The node named `name`.
    virtual Output<ov::Node> get_node(const std::string& name) const;

    /// \brief      Return true if the node named `name` exist in the cache.
    ///
    /// \param[in]  name       The name of the node.
    ///
    /// \return     true if the node named `name` exist in the cache, false otherwise.
    virtual bool contains(const std::string& name) const;

    virtual ~GraphCache() = default;

private:
    std::map<std::string, Output<ov::Node>> m_graph_cache_map;
};
}  // namespace onnx_import
}  // namespace ngraph
