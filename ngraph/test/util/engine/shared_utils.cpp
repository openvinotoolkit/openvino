
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

#include <cmath>
#include <sstream>

#include "shared_utils.hpp"

testing::AssertionResult ngraph::test::compare_with_tolerance(const std::vector<float>& expected,
                                                              const std::vector<float>& results,
                                                              const float tolerance)
{
    auto comparison_result = testing::AssertionSuccess();

    std::stringstream msg;
    msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    bool rc = true;

    for (std::size_t j = 0; j < expected.size(); ++j)
    {
        float diff = std::fabs(results[j] - expected[j]);
        if (diff > tolerance)
        {
            msg << expected[j] << " is not close to " << results[j] << " at index " << j << "\n";
            rc = false;
        }
    }

    if (!rc)
    {
        comparison_result = testing::AssertionFailure();
        comparison_result << msg.str();
    }

    return comparison_result;
}
