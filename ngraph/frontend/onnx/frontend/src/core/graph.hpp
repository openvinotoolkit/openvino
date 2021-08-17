// Copyright (C) 2018-2021 Intel Corporation
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

namespace ngraph {
namespace onnx_import {
class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto);
    Graph() = delete;

    Graph(const Graph&) = delete;
    Graph(Graph&&) = default;

    Graph& operator=(const Graph&) = delete;
    Graph& operator=(Graph&&) = default;
    std::shared_ptr<Function> decode();
    virtual std::shared_ptr<Function> convert();
    OutputVector get_ng_outputs() const;
    const std::string& get_name() const {
        return m_model->get_graph().name();
    }
    const GraphCache& get_graph_cache() const;
    const ParameterVector& get_ng_parameters() const {
        return m_parameters;
    }
    virtual Output<ngraph::Node> get_ng_node_from_cache(const std::string& name) const;
    OutputVector make_ng_nodes(const Node& onnx_node) const;
    const OpsetImports& get_opset_imports() const;
    virtual ~Graph() = default;

protected:
    Graph(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model, std::unique_ptr<GraphCache>&& cache);

    void set_friendly_names(const Node& onnx_node, const OutputVector& ng_node_vector) const;

protected:
    virtual void decode_to_framework_nodes();
    void convert_to_ngraph_nodes();
    void remove_dangling_parameters();
    std::shared_ptr<Function> create_function();

    ParameterVector m_parameters;
    std::unique_ptr<Model> m_model;
    std::unique_ptr<GraphCache> m_cache;

private:
    std::vector<Node> m_nodes;
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
    Subgraph(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model, const Graph& parent_graph);

    /// \brief      Return nodes which are on the edge the subgraph and the parent graph.
    /// \return     Vector of edge nodes from parent scope.
    const std::vector<Output<ngraph::Node>> get_inputs_from_parent() const;

    std::shared_ptr<Function> convert() override;

    Subgraph() = delete;

    Subgraph(const Subgraph&) = delete;
    Subgraph(Subgraph&&) = default;

    Subgraph& operator=(const Subgraph&) = delete;
    Subgraph& operator=(Subgraph&&) = default;

    Output<ngraph::Node> get_ng_node_from_cache(const std::string& name) const override;
    void infer_inputs_from_parent();

private:
    void decode_to_framework_nodes() override;
    void find_inputs_from_parent();

    const GraphCache* m_parent_graph_cache;
    std::vector<std::string> m_inputs_from_parent;
    std::unordered_map<std::shared_ptr<ngraph::op::Parameter>, std::string> m_parameter_to_parent_node_map;
};

inline std::ostream& operator<<(std::ostream& outs, const Graph& graph) {
    return (outs << "<Graph: " << graph.get_name() << ">");
}

}  // namespace onnx_import

}  // namespace ngraph
