// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_place.hpp"

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

JSONTensorPlace::JSONTensorPlace(const ov::frontend::InputModel& input_model, const nlohmann::json& json_data)
    : Place(input_model) {
    if (!json_data.is_object()) {
        FRONT_END_THROW("Invalid JSON data for TensorPlace: not an object");
    }

    // Get the name
    if (json_data.contains("name")) {
        m_name = json_data["name"].get<std::string>();
    } else {
        FRONT_END_THROW("Missing required field 'name' in tensor data");
    }

    // Get input/output flags if present
    m_is_input = json_data.value("is_input", false);
    m_is_output = json_data.value("is_output", false);

    // Get the data type (element type)
    if (json_data.contains("dtype")) {
        const auto& dtype = json_data["dtype"].get<std::string>();
        if (dtype == "float32") {
            m_element_type = ov::element::f32;
        } else if (dtype == "float16") {
            m_element_type = ov::element::f16;
        } else if (dtype == "int64") {
            m_element_type = ov::element::i64;
        } else if (dtype == "int32") {
            m_element_type = ov::element::i32;
        } else {
            FRONT_END_THROW("Unsupported data type: " + dtype);
        }
    } else {
        // Default to float32 if not specified
        m_element_type = ov::element::f32;
    }

    // Get the shape
    if (json_data.contains("shape")) {
        const auto& shape_data = json_data["shape"];
        if (!shape_data.is_array()) {
            FRONT_END_THROW("Invalid shape data: not an array");
        }

        std::vector<ov::Dimension> dims;
        for (const auto& dim : shape_data) {
            if (dim.is_number()) {
                dims.push_back(dim.get<int64_t>());
            } else if (dim.is_string() && dim.get<std::string>() == "-1") {
                // Dynamic dimension
                dims.push_back(ov::Dimension::dynamic());
            } else {
                FRONT_END_THROW("Invalid shape dimension");
            }
        }
        m_partial_shape = ov::PartialShape(dims);
    } else {
        // If no shape is provided, use dynamic shape
        m_partial_shape = ov::PartialShape::dynamic();
    }
}

std::vector<std::string> JSONTensorPlace::get_names() const {
    return {m_name};
}
bool JSONTensorPlace::is_equal(const Place::Ptr& another) const {
    if (auto tensor = std::dynamic_pointer_cast<JSONTensorPlace>(another)) {
        return m_name == tensor->m_name;
    }
    return false;
}

ov::element::Type JSONTensorPlace::get_element_type() const {
    return m_element_type;
}

ov::PartialShape JSONTensorPlace::get_partial_shape() const {
    return m_partial_shape;
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov