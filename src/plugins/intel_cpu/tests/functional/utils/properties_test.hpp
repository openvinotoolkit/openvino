// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"

class OVClassConfigTestCPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;
    const std::string deviceName = "CPU";

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ov::test::utils::make_conv_pool_relu();
    }
};
