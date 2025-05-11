// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/core/shape.hpp"
#include <string>
#include <thread>
#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

// TODO [mandrono]: move current test case inside CPU plug-in and return the original tests
namespace ov {
namespace test {
namespace behavior {

using OVInferRequestDynamicParams = std::tuple<
        std::shared_ptr<Model>,                                         // ov Model
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>,  // input/expected output shapes per inference
        std::string,                                                       // Device name
        ov::AnyMap                                                  // Config
>;

class OVInferRequestDynamicTests : public testing::WithParamInterface<OVInferRequestDynamicParams>,
                                   public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OVInferRequestDynamicParams> obj);

protected:
    void SetUp() override;
    bool checkOutput(const ov::Tensor& in, const ov::Tensor& actual);

    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    std::shared_ptr<Model> function;
    ov::AnyMap configuration;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
};
using OVNotSupportRequestDynamicTests = OVInferRequestDynamicTests;
}  // namespace behavior
}  // namespace test
}  // namespace ov
