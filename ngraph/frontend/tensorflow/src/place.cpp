// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/place.hpp>

using namespace ngraph;
using namespace frontend;

bool PlaceTF::is_input() const {
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PlaceTF::is_output() const {
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}
