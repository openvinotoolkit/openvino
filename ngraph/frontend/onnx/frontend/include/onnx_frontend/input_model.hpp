// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/input_model.hpp>
#include <onnx_editor/editor.hpp>

namespace ngraph
{
    namespace frontend
    {
        class FRONTEND_API InputModelONNX : public InputModel
        {
        public:
            InputModelONNX(const std::string& path);

            std::vector<Place::Ptr> get_inputs() const override;
            Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
            void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) override;
            void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override;

            std::shared_ptr<Function> decode();
            std::shared_ptr<Function> convert();

            // Editor features
            void cut_and_add_new_input(Place::Ptr place, const std::string& new_name_optional = "") override;
            void cut_and_add_new_output(Place::Ptr place, const std::string& new_name_optional = "") override;
            Place::Ptr add_output(Place::Ptr place) override;
            void remove_output(Place::Ptr place) override;
            void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
            void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
            void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;

        private:
            onnx_editor::ONNXModelEditor m_editor;
        };

    } // namespace frontend

} // namespace ngraph
