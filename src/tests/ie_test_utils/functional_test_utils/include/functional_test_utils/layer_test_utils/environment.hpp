// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "ngraph/ngraph.hpp"

#include "functional_test_utils/layer_test_utils/summary.hpp"

namespace LayerTestsUtils {

class TestEnvironment : public ::testing::Environment {
public:
    void TearDown() override {
        Summary::getInstance().saveReport();
    };
};
}  // namespace LayerTestsUtils
