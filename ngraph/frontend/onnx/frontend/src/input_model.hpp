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
        class InputModelONNX : public InputModel
        {
        public:
            InputModelONNX(const std::string& path);

            std::vector<Place::Ptr> get_inputs() const override;
            Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
            void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) override;
            void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override;

            std::shared_ptr<Function> decode();
            std::shared_ptr<Function> convert();

        private:
            onnx_editor::ONNXModelEditor m_editor;
        };

    } // namespace frontend

} // namespace ngraph
