// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <frontend_manager/input_model.hpp>
#include <fstream>

namespace ngraph {
namespace frontend {
class InputModelONNX : public InputModel {
public:
    InputModelONNX(const std::string& path);
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    InputModelONNX(const std::wstring& path);
#endif
    InputModelONNX(std::istream& model_stream);
    // The path can be required even if the model is passed as a stream because it is necessary
    // for ONNX external data feature
    InputModelONNX(std::istream& model_stream, const std::string& path);
    InputModelONNX(std::istream& model_stream, const std::wstring& path);

    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
    Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                          int input_port_index) override;
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) override;
    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override;

    std::shared_ptr<Function> decode();
    std::shared_ptr<Function> convert();

    // Editor features
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;

private:
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

}  // namespace frontend

}  // namespace ngraph
