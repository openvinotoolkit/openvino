// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ov_models/subgraph_builders.hpp"

class OVClassConfigTestCPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;
    const std::string deviceName = "CPU";

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};
