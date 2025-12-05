// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/input_model.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class JSONTensorPlace : public Place {
public:
    JSONTensorPlace(const ov::frontend::InputModel& input_model, const nlohmann::json& json_data);

    virtual ~JSONTensorPlace() = default;

    // Override required virtual methods from Place
    std::vector<std::string> get_names() const override;
    bool is_input() const override {
        return m_is_input;
    }
    bool is_output() const override {
        return m_is_output;
    }
    bool is_equal(const Ptr& another) const override;

    // Additional methods for tensor info
    ov::element::Type get_element_type() const;
    ov::PartialShape get_partial_shape() const;

private:
    std::string m_name;
    bool m_is_input{false};
    bool m_is_output{false};
    ov::element::Type m_element_type;
    ov::PartialShape m_partial_shape;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov