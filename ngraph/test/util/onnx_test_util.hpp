// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ngraph
{
    namespace test
    {
        struct ComparisonResult
        {
            ComparisonResult() = default;
            ComparisonResult(std::string error)
                : is_ok{false}
                , error_message{std::move(error)}
            {
            }
            ComparisonResult(ComparisonResult&&) = default;
            ComparisonResult(const ComparisonResult&) = default;
            ComparisonResult& operator=(ComparisonResult&&) = default;
            ComparisonResult& operator=(const ComparisonResult&) = default;

            bool is_ok = true;
            std::string error_message;

            static ComparisonResult pass() { return {}; }
            static ComparisonResult fail(std::string error)
            {
                return ComparisonResult{std::move(error)};
            }
        };

        ComparisonResult compare_onnx_models(const std::string& model,
                                             const std::string& reference_model_path);

    } // namespace test
} // namespace ngraph
