// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/place.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class Place : public ov::frontend::Place {
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

private:
    const ov::frontend::InputModel& m_input_model;
    const size_t m_tensor_index;
    std::vector<std::string> m_names;
    bool m_is_input;
    bool m_is_output;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
