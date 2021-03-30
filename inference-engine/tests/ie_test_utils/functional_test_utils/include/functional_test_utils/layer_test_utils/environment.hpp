// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pugixml.hpp>
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
