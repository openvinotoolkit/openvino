// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "onnx_editor/editor_types.hpp"

namespace ONNX_NAMESPACE
{
    // Forward declaration to avoid the necessity of including paths in components
    // that don't directly depend on the ONNX library
    class GraphProto;
} // namespace ONNX_NAMESPACE

namespace ngraph
{
    namespace onnx_editor
    {
        /// \brief A class which allows specifying InputEdge and OutputEdge by user-friendly ONNX
        /// names.
        class EdgeMapper
        {
        public:
            ONNX_IMPORTER_API EdgeMapper() = default;

            /// \brief Creates an edge mapper based on a GraphProto object.
            ///
            /// \note If state of graph_proto will be changed, the information from edge mapper
            ///       is outdated. In such a case the update method should be called.
            ///
            /// \param graph_proto Reference to a GraphProto object.
            EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto);

            /// \brief Returns the InputEdge based on a node (node name or output name)
            ///        and an input (input name or input index).
            ///
            /// \note  The node name can be ambiguous (many ONNX nodes can have the same name).
            ///        In such a case the algorthim tries to match the given node name
            ///        with the input name (providing an input index is not enough).
            ///        If a unique edge is found, it will be returned.
            ///        If InputEdge cannot be determined based on parameter values an ngraph_error
            ///        exception will be thrown.
            ///
            /// \param node An EditorNode helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param input An EditorInput helper structure created based on a input name
            ///              or a input index.
            ONNX_IMPORTER_API InputEdge find_input_edge(const EditorNode& node,
                                                        const EditorInput& input) const;

            /// \brief Returns an OutputEdge based on a node (node name or output name)
            ///        and an output (output name or output index).
            ///
            /// \note  The node name can be ambiguous (many ONNX nodes can have the same name).
            ///        In such a case the algorthim will try to match the given node name
            ///        with the output name (providing an output index is not enough).
            ///        If after such operation a found edge is unique, it is returned.
            ///        If OutputEdge cannot be determined based on given params the ngraph_error
            ///        exception is thrown.
            ///
            /// \param node An EditorNode helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param output An EditorOutput helper structure created based on a output name
            ///               or a output index.
            ONNX_IMPORTER_API OutputEdge find_output_edge(const EditorNode& node,
                                                          const EditorOutput& output) const;

            /// \brief Returns an OutputEdge based on a output name.
            ///
            /// \note  The output name guarantees the uniqueness of the edge.
            ///
            /// \param output_name A node output name.
            ///
            ONNX_IMPORTER_API OutputEdge find_output_edge(const std::string& output_name) const;

            /// \brief Returns a vector of InputEdges which consume an output of a node
            ///        determined by provided output name.
            ///
            /// \note  The output name is deterministic in the ONNX standard.
            ///
            /// \param output_name A node output name.
            ///
            ONNX_IMPORTER_API std::vector<InputEdge>
                find_output_consumers(const std::string& output_name) const;

            /// \brief Returns true if a provided node is correct (exists in a graph)
            ///        and is not ambiguous (identification of an ONNX node can be ambiguous
            ///        if an only tensor name is provided).
            ///
            /// \param node An EditorNode helper structure created based on a node name
            ///             or a node output name.
            ///
            ONNX_IMPORTER_API bool is_correct_and_unambiguous_node(const EditorNode& node) const;

        private:
            std::vector<int> find_node_indexes(const std::string& node_name,
                                               const std::string& output_name) const;

            int get_node_input_idx(int node_index, const std::string& input_name) const;
            int get_node_output_idx(int node_index, const std::string& output_name) const;

            std::vector<std::vector<std::string>> m_node_inputs;
            std::vector<std::vector<std::string>> m_node_outputs;
            std::multimap<std::string, int> m_node_name_to_index;
            std::map<std::string, int> m_node_output_name_to_index;
            std::multimap<std::string, int> m_output_consumers_index;
        };
    } // namespace onnx_editor
} // namespace ngraph
