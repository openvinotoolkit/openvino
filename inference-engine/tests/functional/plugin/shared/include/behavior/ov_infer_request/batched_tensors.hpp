// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "ie_extension.h"
#include <condition_variable>
#include "openvino/core/shape.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "transformations/utils/utils.hpp"
#include <string>
#include <ie_core.hpp>
#include <thread>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

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
