// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <fstream>
#include <openvino/frontend/input_model.hpp>

#include "openvino/frontend/extension/holder.hpp"

using ::ONNX_NAMESPACE::ModelProto;

namespace ov {
namespace frontend {
namespace onnx {

class InputModel : public ov::frontend::InputModel {
public:
    InputModel(const std::string& path, const bool enable_mmap = false, ExtensionHolder extensions = {});
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    InputModel(const std::wstring& path, const bool enable_mmap = false, ExtensionHolder extensions = {});
#endif
    InputModel(std::istream& model_stream, const bool enable_mmap = false, ExtensionHolder extensions = {});
    // The path can be required even if the model is passed as a stream because it is necessary
    // for ONNX external data feature
    InputModel(std::istream& model_stream,
               const std::string& path,
               const bool enable_mmap = false,
               ExtensionHolder extensions = {});
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    InputModel(std::istream& model_stream,
               const std::wstring& path,
               const bool enable_mmap = false,
               ExtensionHolder extensions = {});
#endif
    InputModel(std::shared_ptr<ModelProto> model_proto, ExtensionHolder extensions = {});

    std::vector<ov::frontend::Place::Ptr> get_inputs() const override;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const override;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
    ov::frontend::Place::Ptr get_place_by_input_index(size_t input_idx) const override;
    ov::frontend::Place::Ptr get_place_by_operation_name(const std::string& operation_name) const override;
    ov::frontend::Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                        int input_port_index) override;
    ov::frontend::Place::Ptr get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                         int output_port_index) override;

    void set_name_for_tensor(const ov::frontend::Place::Ptr& tensor, const std::string& new_name) override;
    void set_name_for_operation(const ov::frontend::Place::Ptr& operation, const std::string& new_name) override;
    void free_name_for_operation(const std::string& name) override;
    void set_name_for_dimension(const ov::frontend::Place::Ptr& place,
                                size_t shape_dim_index,
                                const std::string& dim_name) override;

    void add_name_for_tensor(const ov::frontend::Place::Ptr& tensor, const std::string& new_name) override;

    /// \brief  Not applicable for ONNX model. Throws immediately
    void free_name_for_tensor(const std::string& name) override;

    void set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape& shape) override;
    ov::PartialShape get_partial_shape(const ov::frontend::Place::Ptr& place) const override;
    void set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type& type) override;
    ov::element::Type get_element_type(const ov::frontend::Place::Ptr& place) const override;
    ov::frontend::Place::Ptr add_output(const ov::frontend::Place::Ptr& place) override;
    void remove_output(const ov::frontend::Place::Ptr& place) override;

    std::shared_ptr<Model> decode();
    std::shared_ptr<Model> convert();

    void cut_and_add_new_input(const ov::frontend::Place::Ptr& place,
                               const std::string& new_name_optional = "") override;

    // Editor features
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs) override;

    // Editor tensor features
    void set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) override;

    // internal usage
    std::vector<InputEdge> convert_place_to_input_edge(const std::vector<ov::frontend::Place::Ptr>& inputs);
    std::vector<OutputEdge> convert_place_to_output_edge(const std::vector<ov::frontend::Place::Ptr>& outputs);

private:
    std::shared_ptr<ONNXModelEditor> m_editor;
    bool is_correct_place(const ov::frontend::Place::Ptr& place) const;

    std::unordered_map<std::string, std::unordered_set<std::string>> m_additional_tensor_names;
    void add_tensor_names(std::shared_ptr<Model>& model);

    std::unordered_map<std::string, ov::PartialShape> m_inputs_to_reshape;
    void reshape_model_inputs(std::shared_ptr<Model>& model);
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
