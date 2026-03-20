// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

bool ov::npuw::util::has_input(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto inputs = model->inputs();
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    return it != inputs.end();
}

