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

#include <gtest/gtest.h>
#include <vector>

namespace ngraph
{
    namespace test
    {
        testing::AssertionResult compare_with_tolerance(const std::vector<float>& expected_results,
                                                        const std::vector<float>& results,
                                                        const float tolerance);
    }
} // namespace ngraph
