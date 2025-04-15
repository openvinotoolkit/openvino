// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"

namespace ov {
namespace test {
namespace op_conformance {

using OpImplParams = std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>;      // Function to check

class OpImplCheckTest : public testing::WithParamInterface<OpImplParams>,
                        public ov::test::TestsCommon {
protected:
    ov::test::utils::OpSummary& summary = ov::test::utils::OpSummary::getInstance();
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function;
    std::string targetDevice;
    ov::AnyMap configuration;

public:
    void SetUp() override;
    static std::string getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj);
};

}   // namespace op_conformance
}   // namespace test
}   // namespace ov
