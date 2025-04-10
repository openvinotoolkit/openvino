// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <memory>

#include "editor_types.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/op/constant.hpp"
#include "utils/tensor_external_data.hpp"

using ::ONNX_NAMESPACE::ModelProto;

namespace ov {
namespace frontend {
namespace onnx {
/// \brief A class representing a set of utilities allowing modification of an ONNX model
///
/// \note This class can be used to modify an ONNX model before it gets translated to
///       an ov::Model by the frontend->convert method. It lets you modify the
///       model's input types and shapes, extract a subgraph and more.
class ONNXModelEditor final {
public:
    /// \brief Creates an editor from a model file located on a storage device. The file
    ///        is parsed and loaded into the m_model_proto member variable.
    ///
    /// \param model_path Path to the file containing the model.
    /// \param enable_mmap Enable mapping files with external weights instead of reading.
    /// \param extensions Holder for custom extensions (like custom ops).
    explicit ONNXModelEditor(const std::string& model_path,
                             const bool enable_mmap = false,
                             frontend::ExtensionHolder extensions = {});
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    ONNXModelEditor(const std::wstring& model_path,
                    const bool enable_mmap = false,
                    frontend::ExtensionHolder extensions = {});
#endif

    /// \brief Creates an editor from a model stream. The stream is parsed and loaded
    ///        into the m_model_proto member variable.
    ///
    /// \param model_stream The stream containing the model.
    /// \param model_path Path to the file containing the model. This information can be used
    ///                   for ONNX external weights feature support.
    /// \param enable_mmap Enable mapping files with external weights instead of reading.
    /// \param extensions Holder for custom extensions (like custom ops).
    explicit ONNXModelEditor(std::istream& model_stream,
                             const std::string& path = {},
                             const bool enable_mmap = false,
                             frontend::ExtensionHolder extensions = {});

    /// \brief Creates an editor from a ModelProto. The model_proto is
    ///        stored in m_model_proto member variable.
    ///
    /// \param model_proto A shared pointer on ModelProto object.
    /// \param extensions Holder for custom extensions (like custom ops).
    ONNXModelEditor(std::shared_ptr<ModelProto> model_proto, frontend::ExtensionHolder extensions = {});

    /// \brief Modifies the in-memory representation of the model by setting
    ///        custom input types for all inputs specified in the provided map.
    ///
    /// \param input_types A collection of pairs {input_name: new_input_type} that should be
    ///                    used to modified the ONNX model loaded from a file. This method
    ///                    throws an exception if the model doesn't contain any of
    ///                    the inputs specified in its parameter.
    void set_input_types(const std::map<std::string, ov::element::Type_t>& input_types);

    /// \brief Modifies the in-memory representation of the model by setting
    ///        custom input shapes for all inputs specified in the provided map.
    ///
    /// \param input_shapes A collection of pairs {input_name: new_input_shape} that should
    ///                     be used to modified the ONNX model loaded from a file. This
    ///                     method throws an exception if the model doesn't contain any of
    ///                     the inputs specified in its parameter.
    void set_input_shapes(const std::map<std::string, ov::PartialShape>& input_shapes);

    /// \brief Get shape of ONNX tensor indicated by the tensor_name.
    ///
    /// \param tensor_name The name of ONNX tensor.
    ///
    PartialShape get_tensor_shape(const std::string& tensor_name) const;

    /// \brief Extracts a subgraph constrained by input edges and output edges. In the end
    ///        the underlying ModelProto is modified - obsolete inputs, initializers, nodes
    ///        and outputs are removed from the in-memory model.
    ///
    /// \note Please look at the declaration of InputEdge and OutputEdge for explanation
    ///       how those objects can be created. If the outputs parameter is empty
    ///       this method keeps all of the original outputs of the model.
    ///
    /// \param inputs A collection of input edges which become new inputs to the graph
    /// \param outputs A collection of output edges which become new outputs of the graph
    /// \param merge_inputs Flag indicates whether newly created inputs after cutting shall be independent or merged,
    ///                     false - each cutted edge will be connected with one new input (default),
    ///                     true - all input edges will be connected to one new input
    void extract_subgraph(const std::vector<InputEdge>& inputs,
                          const std::vector<OutputEdge>& outputs,
                          const bool merge_inputs = false);

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
    void set_input_values(const std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>& input_values);

    /// \brief Changes the name of given tensor.
    ///
    /// \note It changes input, output, initializer and value_info proto repeated fields as well as
    ///       all nodes which refer to the tensor.
    ///
    /// \param current_name Name of tensor to be changed.
    /// \param new_name New name of tensor. Must not be empty nor point to existing tensor (including self).
    void set_tensor_name(const std::string& current_name, const std::string& new_name);

    /// \brief Sets node's name.
    ///
    /// \note Empty name is accepted.
    ///
    /// \param node Handle to node.
    /// \param new_name New name of the node.
    void set_node_name(const EditorNode& node, const std::string& new_name);

    /// \brief Retrieves a node name from the in-memory ONNX model.
    ///
    /// \param node Node descriptor for which the lookup should be performed.
    std::string get_node_name(const EditorNode& node) const;

    /// \brief Removes node name for all nodes with given name.
    ///
    /// \note Empty and not present names are accepted.
    ///
    /// \param name Name to clear
    void clear_nodes_name(const std::string& name);

    /// \brief Overrides or creates name for tensor shape dimension (numeric dimension is erased).
    ///
    /// \note It changes input, output and value_info proto repeated fields.
    ///       If rank of the tensor is too low the shape is expanded with dynamic dimensions so
    ///       the name can be set at specified position.
    ///
    /// \param node_name Tensor name to change its shape. Must not point to initializer.
    /// \param shape_dim_index Index of dimension to change.
    /// \param dim_name New name of the dimension. Must not be empty.
    void set_name_for_dimension(const std::string& node_name, size_t shape_dim_index, const std::string& dim_name);

    /// \brief Returns a serialized ONNX model, possibly modified by the editor.
    std::string model_string() const;

    /// \brief     Converts an edited ONNX model to an OpenVINO Model representation.
    std::shared_ptr<Model> get_function() const;

    /// \brief Returns a list of all inputs of the in-memory model.
    ///        The returned value might depend on the previous operations executed on an
    ///        instance of the model editor, in particular the subgraph extraction which
    ///        can discard some inputs from the original graph.
    ///
    ///  \note ONNX initializers is not treated as input of the model.
    std::vector<std::string> model_inputs() const;

    /// \brief Returns a list of all outputs of the in-memory model.
    ///        The returned value might depend on the previous operations executed on an
    ///        instance of the model editor.
    std::vector<std::string> model_outputs() const;

    /// \brief     Get name of the tensor which is the source of the input edge.
    ///
    /// \note      Empty string is returned if the tensor name is not found.
    ///
    std::string get_source_tensor_name(const InputEdge& edge) const;

    /// \brief     Returns true if input edge is input of the model. Otherwise false.
    bool is_input(const InputEdge& edge) const;

    /// \brief     Get name of the tensor which is the target of the output edge.
    ///
    /// \note      Empty string is returned if the tensor name is not found.
    ///
    std::string get_target_tensor_name(const OutputEdge& edge) const;

    /// \brief     Returns true if output edge is output of the model. Otherwise false.
    bool is_output(const OutputEdge& edge) const;

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
    ///        If InputEdge cannot be determined based on parameter values an ov::Exception
    ///        will be thrown.
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
    ///        If OutputEdge cannot be determined based on given params the ov::Exception
    ///        will be thrown.
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

    /// \brief Returns index (position) of provided node in the graph
    ///        in topological order.
    ///
    /// \param node An EditorNode helper structure created based on a node name
    ///             or a node output name.
    ///
    /// \note  The exception will be thrown if the provided node is ambiguous.
    ///
    int get_node_index(const EditorNode& node) const;

    /// \brief Returns true if a provided tensor name is correct (exists in a graph).
    ///
    /// \param name The name of tensor in a graph.
    ///
    bool is_correct_tensor_name(const std::string& name) const;

    /// \brief     Get names of input ports of given node.
    ///
    /// \param node An EditorNode helper structure created based on a node name
    ///             or a node output name.
    ///
    std::vector<std::string> get_input_ports(const EditorNode& node) const;

    /// \brief     Get names of output ports of given node.
    ///
    /// \param node An EditorNode helper structure created based on a node name
    ///             or a node output name.
    ///
    std::vector<std::string> get_output_ports(const EditorNode& node) const;

    /// \brief Returns a OpenVINO Model based on edited model
    ///        decoded to framework nodes
    ///
    std::shared_ptr<Model> decode();

    /// \brief     Adds output to provided OutputEdge.
    ///
    /// \param output_edge An output_edge type where graph output shall be added.
    ///
    void add_output(const OutputEdge& output_edge) const;

    /// \brief     Provides element type for given input tensor name.
    ///
    /// \param output_edge Name of tensor for which element type will be returned.
    ///
    ov::element::Type_t get_input_type(const std::string& tensor_name) const;

private:
    void update_mapper_if_needed() const;

    const std::string m_model_path;
    ov::frontend::onnx::detail::MappedMemoryHandles m_mmap_cache;
    frontend::ExtensionHolder m_extensions;

    struct Impl;
    std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
