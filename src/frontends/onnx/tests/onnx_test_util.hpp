// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>

namespace ngraph {
namespace test {
struct ComparisonResult {
    ComparisonResult() = default;
    ComparisonResult(std::string error) : is_ok{false}, error_message{std::move(error)} {}
    ComparisonResult(ComparisonResult&&) = default;
    ComparisonResult(const ComparisonResult&) = default;
    ComparisonResult& operator=(ComparisonResult&&) = default;
    ComparisonResult& operator=(const ComparisonResult&) = default;

    bool is_ok = true;
    std::string error_message;

    static ComparisonResult pass() {
        return {};
    }
    static ComparisonResult fail(std::string error) {
        return ComparisonResult{std::move(error)};
    }
};

bool default_name_comparator(std::string lhs, std::string rhs);

// comp is a function to compare inputs and outputs names (as default it is a usual std::string comparison)
using CompType = std::function<bool(std::string, std::string)>;
ComparisonResult compare_onnx_models(const std::string& model,
                                     const std::string& reference_model_path,
                                     CompType comp = default_name_comparator);

}  // namespace test
}  // namespace ngraph
