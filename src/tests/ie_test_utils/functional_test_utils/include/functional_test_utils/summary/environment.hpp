// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "ngraph/ngraph.hpp"

#include "functional_test_utils/summary/op_summary.hpp"
#include "functional_test_utils/summary/api_summary.hpp"

namespace ov {
namespace test {
namespace utils {

class TestEnvironment : public ::testing::Environment {
public:
    void TearDown() override {
#ifdef ENABLE_API_REPORT
        OpSummary::getInstance().saveReport();
#endif
        ApiSummary::getInstance().saveReport();
    };
};

}  // namespace utils
}  // namespace test
}  // namespace ov
