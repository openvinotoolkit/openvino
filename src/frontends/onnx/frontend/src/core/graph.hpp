// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <memory>
#include <string>
#include <vector>

#include "core/graph_cache.hpp"
#include "core/model.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "onnx_import/core/operator_set.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "ops_bridge.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph {
namespace onnx_import {
class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph(const std::string& model_dir,
          const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
          detail::MappedMemoryHandles mmap_cache,
          ov::frontend::ExtensionHolder extensions = {});
    Graph() = delete;

    Graph(const Graph&) = delete;
    Graph(Graph&&) = default;

    Graph& operator=(const Graph&) = delete;
    Graph& operator=(Graph&&) = default;
    std::shared_ptr<Function> decode();
    virtual std::shared_ptr<Function> convert();
    OutputVector get_ng_outputs();
    const std::string& get_name() const {
        return m_model->get_graph().name();
    }
    const std::string& model_dir() const {
        return m_model_dir;
    }
    detail::MappedMemoryHandles get_mmap_cache() const {
        return m_mmap_cache;
    }
    const ParameterVector& get_ng_parameters() const {
        return m_parameters;
    }
    virtual bool is_ng_node_in_cache(const std::string& name) const;
    virtual Output<ngraph::Node> get_ng_node_from_cache(const std::string& name);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OutputVector make_ng_nodes(const Node& onnx_node);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const OpsetImports& get_opset_imports() const;
    virtual ~Graph() = default;

    const ov::frontend::ExtensionHolder& get_extensions() const {
        return m_extensions;
    }

protected:
    Graph(const std::string& model_dir,
          const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model,
          std::unique_ptr<GraphCache>&& cache,
          detail::MappedMemoryHandles mmap_cache,
          ov::frontend::ExtensionHolder extensions = {});

    OPENVINO_SUPPRESS_DEPRECATED_START
    void set_friendly_names(const Node& onnx_node, const OutputVector& ng_subgraph_outputs) const;
    OPENVINO_SUPPRESS_DEPRECATED_END

protected:
    OPENVINO_SUPPRESS_DEPRECATED_START
    OutputVector make_framework_nodes(const Node& onnx_node);
    OPENVINO_SUPPRESS_DEPRECATED_END
    void decode_to_framework_nodes();
    void convert_to_ngraph_nodes();
    void remove_dangling_parameters();
    void set_metadata(std::shared_ptr<ov::Model>& model) const;
    std::shared_ptr<Function> create_function();

    ParameterVector m_parameters;
    std::unique_ptr<Model> m_model;
    std::unique_ptr<GraphCache> m_cache;
    ov::frontend::ExtensionHolder m_extensions = {};

private:
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::vector<Node> m_nodes;
    OPENVINO_SUPPRESS_DEPRECATED_END
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
    Subgraph(const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model, Graph* parent_graph);

    /// \brief      Return nodes which are on the edge the subgraph and the parent graph.
    /// \return     Vector of edge nodes from parent scope.
    const std::vector<Output<ngraph::Node>> get_inputs_from_parent() const;

    std::shared_ptr<Function> convert() override;

    Subgraph() = delete;

    Subgraph(const Subgraph&) = delete;
    Subgraph(Subgraph&&) = default;

    Subgraph& operator=(const Subgraph&) = delete;
    Subgraph& operator=(Subgraph&&) = default;

    bool is_ng_node_in_cache(const std::string& name) const override;
    Output<ngraph::Node> get_ng_node_from_cache(const std::string& name) override;
    void infer_inputs_from_parent();

private:
    Graph* m_parent_graph;
    std::vector<std::string> m_inputs_from_parent;
    std::unordered_map<std::shared_ptr<ngraph::op::Parameter>, std::string> m_parameter_to_parent_node_map;
};

inline std::ostream& operator<<(std::ostream& outs, const Graph& graph) {
    return (outs << "<Graph: " << graph.get_name() << ">");
}

static const char* const ONNX_GRAPH_RT_ATTRIBUTE = "onnx_graph";

}  // namespace onnx_import

}  // namespace ngraph
