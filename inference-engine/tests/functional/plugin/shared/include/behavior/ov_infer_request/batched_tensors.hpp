// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include <base/behavior_test_utils.hpp>

namespace ov {
namespace test {
namespace behavior {

class OVInferRequestBatchedTests : public testing::WithParamInterface<std::string>,
                                   public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& device_name);

protected:
    void SetUp() override;

    void TearDown() override;

    std::shared_ptr<runtime::Core> ie = utils::PluginCache::get().core();
    std::string targetDevice;
};
}  // namespace behavior
}  // namespace test
}  // namespace ov
