// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <Configuration.h>
#include <args_parser.h>

class RegisterConfigTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:
    RegisterConfigTests() : Test() {
    }

    virtual ~RegisterConfigTests() {
    }
};

REGISTER_STRING_PARAM(model, m, "!", "someModelMessage", MODEL);
REGISTER_STRING_PARAM(plugin, p, "", "somePluginMessage", PLUGIN);
REGISTER_UINT32_PARAM(niter, ni, 3, "someIterNumMessage", ITER_NUM);
REGISTER_BOOL_PARAM(perf_count, pc, false, "somePerfCountMessage", PERF_COUNT);

TEST_F(RegisterConfigTests, canRegisterParams) {
    Configuration config;
    RegisterConfig::RegisterConfigBinding::deploy(config);
    ASSERT_STREQ(config.value<MODEL>().c_str(), "!");
    ASSERT_STREQ(config.value<PLUGIN>().c_str(), "");
    ASSERT_EQ(config.value<ITER_NUM>(), 3);
    ASSERT_FALSE(config.value<PERF_COUNT>());
}

