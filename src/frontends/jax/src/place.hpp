// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/place.hpp"

namespace ov {
namespace frontend {
namespace jax {

class InputModel;

// Place represents a tensor in the model
class Place : public ov::frontend::Place {
    friend class ::ov::frontend::jax::InputModel;

public:
    Place(const ov::frontend::InputModel& input_model, size_t tensor_index);

    ~Place() override = default;

    bool is_input() const override {
        return m_is_input;
    }
    bool is_output() const override {
        return m_is_output;
    }
    bool is_equal(const Ptr& another) const override {
        return this == another.get();
    }
    std::vector<std::string> get_names() const override {
        return m_names;
    }
    size_t get_tensor_index() const {
        return m_tensor_index;
    }
    element::Type get_element_type() const {
        return m_type;
    }
    PartialShape get_partial_shape() const {
        return m_pshape;
    }
    bool is_equal_data(const Ptr& another) const override {
        const auto another_pt = dynamic_cast<ov::frontend::jax::Place*>(another.get());
        if (!another_pt) {
            return false;
        }
        return m_tensor_index == another_pt->get_tensor_index();
    }

private:
    const ov::frontend::InputModel& m_input_model;
    const size_t m_tensor_index;
    std::vector<std::string> m_names;
    element::Type m_type = element::dynamic;
    PartialShape m_pshape;
    bool m_is_input = false;
    bool m_is_output = false;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
