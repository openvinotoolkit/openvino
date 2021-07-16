// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_editor/editor.hpp"
#include "onnx_editor/editor_types.hpp"

namespace ngraph
{
    namespace onnx_editor
    {
        /// \brief A class representing a set of utilities allowing modification of an ONNX model
        ///
        /// \note This class can be used to modify an ONNX model before it gets translated to
        ///       an ngraph::Function by the import_onnx_model function. It lets you modify the
        ///       model's input types and shapes, extract a subgraph and more.
        class ONNX_IMPORTER_API ONNXModelEditor final
        {
        public:
            ONNXModelEditor() = delete;

            /// \brief Creates an editor from a model file located on a storage device. The file
            ///        is parsed and loaded into the m_model_proto member variable.
            ///
            /// \param model_path Path to the file containing the model.
            ONNXModelEditor(const std::string& model_path);

            /// \brief Modifies the in-memory representation of the model by setting
            ///        custom input types for all inputs specified in the provided map.
            ///
            /// \param input_types A collection of pairs {input_name: new_input_type} that should be
            ///                    used to modified the ONNX model loaded from a file. This method
            ///                    throws an exception if the model doesn't contain any of
            ///                    the inputs specified in its parameter.
            void set_input_types(const std::map<std::string, element::Type_t>& input_types);

            /// \brief Modifies the in-memory representation of the model by setting
            ///        custom input shapes for all inputs specified in the provided map.
            ///
            /// \param input_shapes A collection of pairs {input_name: new_input_shape} that should
            ///                     be used to modified the ONNX model loaded from a file. This
            ///                     method throws an exception if the model doesn't contain any of
            ///                     the inputs specified in its parameter.
            void set_input_shapes(const std::map<std::string, ngraph::PartialShape>& input_shapes);

            /// \brief Extracts a subgraph constrained by input edges and output edges. In the end
            ///        the underlying ModelProto is modified - obsolete inputs, initializers, nodes
            ///        and outputs are removed from the in-memory model.
            ///
            /// \node Please look at the declaration of InputEdge and OutputEdge for explanation
            ///       how those objects can be created. If the outputs parameter is empty
            ///       this method keeps all of the original outputs of the model.
            ///
            /// \param inputs A collection of input edges which become new inputs to the graph
            /// \param outputs A collection of output edges which become new outputs of the graph
            void cut_graph_fragment(const std::vector<InputEdge>& inputs,
                                    const std::vector<OutputEdge>& outputs);

            /// \brief Modifies the in-memory representation of the model by setting custom input
            ///        values for inputs specified in the provided map.
            ///
            /// \note This method modifies existing initializer tensor if its name matches one of
            ///       input_name. Otherwise it adds initializer tensor into the model.
            ///       If input tensor of matching name is present in the model, its type and shape
            ///       are modified accordingly.
            ///
            /// \param input_values A collection of pairs {input_name: new_input_values} used to
            ///                     update the ONNX model. Initializers already existing are
            ///                     overwritten.
            void set_input_values(
                const std::map<std::string, std::shared_ptr<ngraph::op::Constant>>& input_values);

            /// \brief Returns a serialized ONNX model, possibly modified by the editor.
            std::string model_string() const;

            /// \brief     Converts an edited ONNX model to an nGraph Function representation.
            std::shared_ptr<Function> get_function() const;

            /// \brief Returns a list of all inputs of the in-memory model, including initializers.
            ///        The returned value might depend on the previous operations executed on an
            ///        instance of the model editor, in particular the subgraph extraction which
            ///        can discard some inputs and initializers from the original graph.
            std::vector<std::string> model_inputs() const;

            /// \brief Returns the path to the original model file
            const std::string& model_path() const;

            /// \brief Saves the possibly modified model held by this class to a file.
            /// Serializes in binary mode.
            ///
            /// \param out_file_path A path to the file where the modified model should be dumped.
            void serialize(const std::string& out_file_path) const;

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
            /// \param node A node helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param input An input helper structure created based on a input name
            ///              or a input index.
            InputEdge find_input_edge(const EditorNode& node, const EditorInput& input) const;

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
            /// \param node A node helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param output A output helper structure created based on a output name
            ///               or a output index.
            OutputEdge find_output_edge(const EditorNode& node, const EditorOutput& output) const;

            /// \brief Returns an OutputEdge based on a output name.
            ///
            /// \note  The output name guarantees the uniqueness of the edge.
            ///
            /// \param output_name A node output name.
            ///
            OutputEdge find_output_edge(const std::string& output_name) const;

            /// \brief Returns a vector of InputEdges which consume an output of a node
            ///        determined by provided output name.
            ///
            /// \note  The output name is deterministic in the ONNX standard.
            ///
            /// \param output_name A node output name.
            ///
            std::vector<InputEdge> find_output_consumers(const std::string& output_name) const;

            /// \brief Returns a vector of InputEdges which consume an output of a node
            ///        determined by provided output name.
            ///
            /// \note  The output name is deterministic in the ONNX standard.
            ///
            /// \param output_name A node output name.
            ///
            bool is_correct_and_unambiguous_node(const EditorNode& node) const;

        private:
            void update_mapper_if_needed() const;

            const std::string m_model_path;

            struct Impl;
            std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
        };
    } // namespace onnx_editor
} // namespace ngraph
