// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/ov_behavior_test_utils.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "openvino/core/version.hpp"

namespace ov {
namespace test {
namespace behavior {

using BlobPair = std::pair<const char*, const char*>;

class OVBlobCompatibilityNPU : public OVCompiledNetworkTestBase, public testing::WithParamInterface<BlobMap> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, configuration) = this->GetParam();
        if (NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM != configuration.first) {
            GTEST_SKIP();
        }
        APIBaseTest::SetUp();
    }

protected:
    BlobPair configuration;
};

TEST_P(OVBlobCompatibilityNPU, CheckBlobsWithDifferentVersionsAreCompatible) {
    ov::Core;
    OV_ASSERT_NO_THROW(core.import_model(blob, target_devivce, {}));
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
