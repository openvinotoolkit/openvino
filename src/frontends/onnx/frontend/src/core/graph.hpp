// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <memory>
#include <string>
#include <vector>

#include "core/graph_cache.hpp"
#include "core/model.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "openvino/op/parameter.hpp"
#include "ops_bridge.hpp"
#include "utils/tensor_external_data.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph(const std::string& model_dir,
          const std::shared_ptr<ModelProto>& model_proto,
          detail::MappedMemoryHandles mmap_cache,
          ov::frontend::ExtensionHolder extensions = {});
    Graph() = delete;

    Graph(const Graph&) = delete;
    Graph(Graph&&) = default;

    Graph& operator=(const Graph&) = delete;
    Graph& operator=(Graph&&) = default;
    std::shared_ptr<ov::Model> decode();
    virtual std::shared_ptr<ov::Model> convert();
    ov::OutputVector get_ov_outputs();
    const std::string& get_name() const {
        return m_model->get_graph().name();
    }
    const std::string& model_dir() const {
        return m_model_dir;
    }
    detail::MappedMemoryHandles get_mmap_cache() const {
        return m_mmap_cache;
    }
    const ov::ParameterVector& get_ng_parameters() const {
        return m_parameters;
    }
    virtual bool is_ov_node_in_cache(const std::string& name) const;
    virtual ov::Output<ov::Node> get_ov_node_from_cache(const std::string& name);

    ov::OutputVector make_ov_nodes(const ov::frontend::onnx::Node& onnx_node);

    const OpsetImports& get_opset_imports() const;
    virtual ~Graph() = default;

    const ov::frontend::ExtensionHolder& get_extensions() const {
        return m_extensions;
    }

protected:
    Graph(const std::string& model_dir,
          const std::shared_ptr<ModelProto>& model,
          std::unique_ptr<GraphCache>&& cache,
          detail::MappedMemoryHandles mmap_cache,
          ov::frontend::ExtensionHolder extensions = {});

    void set_friendly_names(const Node& onnx_node, const ov::OutputVector& ng_subgraph_outputs) const;

protected:
    ov::OutputVector make_framework_nodes(const ov::frontend::onnx::Node& onnx_node);

    void decode_to_framework_nodes();
    void convert_to_ov_nodes();
    void remove_dangling_parameters();
    void set_metadata(std::shared_ptr<ov::Model>& model) const;
    std::shared_ptr<ov::Model> create_model();

    ov::ParameterVector m_parameters;
    std::unique_ptr<Model> m_model;
    std::unique_ptr<GraphCache> m_cache;
    ov::frontend::ExtensionHolder m_extensions = {};

private:
    std::vector<Node> m_nodes;

    std::string m_model_dir;
    detail::MappedMemoryHandles m_mmap_cache;
    OperatorsBridge m_ops_bridge;
};

/// \brief      Representation of ONNX subgraph. It is used for example by ONNX Loop op.
///             It has access for initializers both from subgraph and from parent graph
///             cache.
class Subgraph : public Graph {
public:
    /// \brief      Subgraph a GraphCache class object.
    ///
    /// \param[in]  model          The ONNX model object.
    /// \param[in]  parent_graph   The reference to the parent graph.
    Subgraph(const std::shared_ptr<ModelProto>& model, Graph* parent_graph);

    /// \brief      Return nodes which are on the edge the subgraph and the parent graph.
    /// \return     Vector of edge nodes from parent scope.
    const std::vector<ov::Output<ov::Node>> get_inputs_from_parent() const;

    std::shared_ptr<ov::Model> convert() override;

    Subgraph() = delete;

    Subgraph(const Subgraph&) = delete;
    Subgraph(Subgraph&&) = default;

    Subgraph& operator=(const Subgraph&) = delete;
    Subgraph& operator=(Subgraph&&) = default;

    bool is_ov_node_in_cache(const std::string& name) const override;
    ov::Output<ov::Node> get_ov_node_from_cache(const std::string& name) override;
    void infer_inputs_from_parent();

private:
    Graph* m_parent_graph;
    std::vector<std::string> m_inputs_from_parent;
    std::unordered_map<std::shared_ptr<ov::op::v0::Parameter>, std::string> m_parameter_to_parent_node_map;
};

inline std::ostream& operator<<(std::ostream& outs, const Graph& graph) {
    return (outs << "<Graph: " << graph.get_name() << ">");
}

static const char* const ONNX_GRAPH_RT_ATTRIBUTE = "onnx_graph";

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
