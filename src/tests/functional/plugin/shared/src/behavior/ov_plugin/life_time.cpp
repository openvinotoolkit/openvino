// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <ngraph_functions/subgraph_builders.hpp>
#include <base/behavior_test_utils.hpp>
#include "behavior/ov_plugin/life_time.hpp"

namespace ov {
namespace test {
namespace behavior {
std::string OVHoldersTest::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    return "targetDevice=" + obj.param;
}

void OVHoldersTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    targetDevice = this->GetParam();
    deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
    if (deathTestStyle == "fast") {
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    }
    function = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVHoldersTest::TearDown() {
    ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
}

#define EXPECT_NO_CRASH(_statement) \
EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

static void release_order_test(std::vector<std::size_t> order, const std::string &deviceName,
                               std::shared_ptr<ngraph::Function> function) {
    ov::AnyVector objects;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, deviceName);
        auto request = compiled_model.create_infer_request();

        objects = {core, compiled_model, request};
    }
    for (auto&& i : order) {
        objects.at(i) = {};
    }
}

TEST_P(OVHoldersTest, Orders) {
    std::vector<std::string> objects{ "core", "compiled_model", "request"};
    std::vector<std::size_t> order(objects.size());
    std::iota(order.begin(), order.end(), 0);
    do {
        std::stringstream order_str;
        for (auto&& i : order) {
            order_str << objects.at(i) << " ";
        }
        EXPECT_NO_CRASH(release_order_test(order, targetDevice, function)) << "for order: " << order_str.str();
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST_P(OVHoldersTest, LoadedState) {
    std::vector<ov::VariableState> states;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, targetDevice);
        auto request = compiled_model.create_infer_request();
        try {
            states = request.query_state();
        } catch(...) {}
    }
}

TEST_P(OVHoldersTest, LoadedTensor) {
    ov::Tensor tensor;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, targetDevice);
        auto request = compiled_model.create_infer_request();
        tensor = request.get_input_tensor();
    }
}

TEST_P(OVHoldersTest, LoadedAny) {
    ov::Any any;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, targetDevice);
        any = compiled_model.get_property(ov::supported_properties.name());
    }
}

TEST_P(OVHoldersTest, LoadedRemoteContext) {
    ov::RemoteContext ctx;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, targetDevice);
        try {
            ctx = compiled_model.get_context();
        } catch(...) {}
    }
}


std::string OVHoldersTestOnImportedNetwork::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    return "targetDevice=" + obj.param;
}

void OVHoldersTestOnImportedNetwork::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    targetDevice = this->GetParam();
    deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
    if (deathTestStyle == "fast") {
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    }
    function = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVHoldersTestOnImportedNetwork::TearDown() {
    ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
}

TEST_P(OVHoldersTestOnImportedNetwork, LoadedTensor) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, targetDevice);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, targetDevice);
    auto request = compiled_model.create_infer_request();
    ov::Tensor tensor = request.get_input_tensor();
}

TEST_P(OVHoldersTestOnImportedNetwork, CreateRequestWithCoreRemoved) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, targetDevice);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, targetDevice);
    core = ov::Core{};
    auto request = compiled_model.create_infer_request();
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
