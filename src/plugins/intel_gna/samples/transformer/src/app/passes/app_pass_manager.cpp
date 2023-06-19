// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "app_pass_manager.hpp"

#include "insert_add_after_parameter.hpp"

namespace transformation_sample {
namespace passes {

AppPassManager::UnknownPassNameException::UnknownPassNameException(const std::string& message)
    : std::invalid_argument(message) {}

std::vector<std::string> AppPassManager::available_passes_names() {
    auto& passes = get_passes();

    std::vector<std::string> names;
    std::transform(passes.begin(), passes.end(), std::back_inserter(names), [](const PassEntry& pass_entry) {
        return pass_entry.first;
    });
    return names;
}

std::shared_ptr<ov::pass::PassBase> AppPassManager::get_pass(const std::string& pass_name) {
    auto& passes = get_passes();
    auto pass_iter = passes.find(pass_name);
    if (pass_iter == passes.end()) {
        std::string error_message = "Error: Received Unknown pass name: ";
        error_message.append(pass_name);
        throw UnknownPassNameException(error_message);
    }

    return pass_iter->second;
}

const AppPassManager::Passes& AppPassManager::get_passes() {
    static const Passes passes = {{"InsertAddAfterParameter", std::make_shared<InsertAddAfterParameter>()}};
    return passes;
}

}  // namespace passes
}  // namespace transformation_sample