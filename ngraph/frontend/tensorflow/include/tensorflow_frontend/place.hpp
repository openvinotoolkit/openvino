// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>

namespace ngraph {
namespace frontend {

class PlaceTF : public Place {
public:
    PlaceTF(const InputModel& input_model, const std::vector<std::string>& names)
        : m_input_model(input_model),
          m_names(names) {}

    explicit PlaceTF(const InputModel& input_model) : PlaceTF(input_model, std::vector<std::string>{}) {}

    ~PlaceTF() override = default;

    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(Ptr another) const override {
        return this == another.get();
    }

    std::vector<std::string> get_names() const override {
        return m_names;
    }

private:
    const InputModel& m_input_model;
    std::vector<std::string> m_names;
};

}  // namespace frontend
}  // namespace ngraph
