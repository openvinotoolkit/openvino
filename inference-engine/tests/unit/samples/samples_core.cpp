// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "../../tests/performance/core/Configuration.h"

class SampleCoreTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:
    SampleCoreTests() : Test() {
    }

    virtual ~SampleCoreTests() {
    }
};


TEST_F(SampleCoreTests, canSetValue) {
    Configuration config;
    config.setValue<MODEL>("Model");
    ASSERT_STREQ(config.value<MODEL>().c_str(), "Model");
    ASSERT_FALSE(config.value<MODEL>().empty());
}

TEST_F(SampleCoreTests, canAddImages) {
    Configuration config;
    std::vector<std::string>& images = config.value<IMAGES>();
    images.push_back("smth");

    ASSERT_FALSE(config.value<IMAGES>().empty());
}

TEST_F(SampleCoreTests, canCreateConstConfig) {
    Configuration config;
    config.value<PLUGIN_PATHS>() = { "plugin_paths" };
    config.value<MODEL>() = "Model";
    config.value<PLUGIN>() = "Plugin";
    std::string path = config.value<PLUGIN_PATHS>().at(0);

    const Configuration config2 = config;

    ASSERT_FALSE(config2.value<PLUGIN_PATHS>().empty());
    ASSERT_FALSE(config.value<MODEL>().empty());
    ASSERT_STREQ(config.value<PLUGIN>().c_str(), "Plugin");
}
