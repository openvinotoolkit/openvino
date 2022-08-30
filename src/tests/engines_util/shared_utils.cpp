// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_utils.hpp"

#include <cmath>
#include <sstream>

testing::AssertionResult ngraph::test::compare_with_tolerance(const std::vector<float>& expected,
                                                              const std::vector<float>& results,
                                                              const float tolerance) {
    auto comparison_result = testing::AssertionSuccess();

    std::stringstream msg;
    msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    bool rc = true;

    for (std::size_t j = 0; j < expected.size(); ++j) {
        float diff = std::fabs(results[j] - expected[j]);
        if (diff > tolerance) {
            msg << expected[j] << " is not close to " << results[j] << " at index " << j << "\n";
            rc = false;
        }
    }

    if (!rc) {
        comparison_result = testing::AssertionFailure();
        comparison_result << msg.str();
    }

    return comparison_result;
}
