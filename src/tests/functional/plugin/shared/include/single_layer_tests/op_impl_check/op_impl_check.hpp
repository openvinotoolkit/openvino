// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"

#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

using OpImplParams = std::tuple<
        std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>,       // Function to check
        std::string,                         // Target Device
        ov::AnyMap>; // Plugin Config

class OpImplCheckTest : public testing::WithParamInterface<OpImplParams>,
                        public CommonTestUtils::TestsCommon {
protected:
    LayerTestsUtils::Summary& summary = LayerTestsUtils::Summary::getInstance();
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function;
    std::string targetDevice;
    ov::AnyMap configuration;

public:
    void run();
    void SetUp() override;
    static std::string getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj);
};

}   // namespace subgraph
}   // namespace test
}   // namespace ov
