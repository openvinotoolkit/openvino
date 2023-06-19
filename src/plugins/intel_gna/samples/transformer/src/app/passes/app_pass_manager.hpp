// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/pass.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace transformation_sample {
namespace passes {

class AppPassManager {
public:
    class UnknownPassNameException : public std::invalid_argument {
    public:
        UnknownPassNameException(const std::string& message);
    };

    std::vector<std::string> available_passes_names();

    std::shared_ptr<ov::pass::PassBase> get_pass(const std::string& pass_name);

private:
    using Passes = std::unordered_map<std::string, std::shared_ptr<ov::pass::PassBase>>;
    using PassEntry = Passes::value_type;
    static const Passes& get_passes();
};
}  // namespace passes
}  // namespace transformation_sample
