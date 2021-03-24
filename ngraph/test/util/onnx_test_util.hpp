//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
