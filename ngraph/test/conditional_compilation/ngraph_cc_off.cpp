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

#include "gtest/gtest.h"

#include <ngraph/except.hpp>

#ifdef SELECTIVE_BUILD_ANALYZER
#define SELECTIVE_BUILD_ANALYZER_ON
#undef SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD)
#define SELECTIVE_BUILD_ON
#undef SELECTIVE_BUILD
#endif

#include "../core/src/itt.hpp"

using namespace std;

TEST(conditional_compilation, op_scope_with_disabled_cc)
{
    int n = 0;

    // Simple scope is enabled
    NGRAPH_OP_SCOPE(Scope0);
    n = 42;
    EXPECT_EQ(n, 42);

    // Simple scope is disabled
    NGRAPH_OP_SCOPE(Scope1);
    n = 43;
    EXPECT_EQ(n, 43);
}

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#define SELECTIVE_BUILD
#endif
