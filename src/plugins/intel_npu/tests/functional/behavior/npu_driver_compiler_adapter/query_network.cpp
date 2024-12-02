// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/op.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu_non_zero.hpp"
#include "intel_npu/config/common.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {

class UnsupportedTestOp : public ov::op::Op {
public:
    OPENVINO_OP("UnsupportedTestOp");

    UnsupportedTestOp() = default;
    explicit UnsupportedTestOp(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_pshape = get_input_partial_shape(0);
        auto input_shape = input_pshape.to_shape();
        ov::Shape output_shape(input_shape);
        set_output_type(0, get_input_element_type(0), ov::PartialShape(output_shape));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            OPENVINO_THROW("Incorrect number of new arguments");
        }

        return std::make_shared<UnsupportedTestOp>(new_args.at(0));
    }
};

std::shared_ptr<ov::Model> createModelWithUnknownNode();

std::shared_ptr<ov::Model> createModelWithUnknownNode() {
    const ov::Shape input_shape = {1, 4096};
    const ov::element::Type precision = ov::element::f32;
    ov::ParameterVector params = {std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{input_shape})};
    auto constant = ov::test::utils::make_constant(precision, ov::Shape{4096, 1024});
    auto custom_op = std::make_shared<UnsupportedTestOp>(constant);

    ov::NodeVector results{custom_op};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "CustomOpModel");
}

class QueryNetworkTestNPU : public ov::test::behavior::OVPluginTestBase,
                            public testing::WithParamInterface<CompilationParams> {
public:
    ov::SupportedOpsMap testQueryNetwork(std::shared_ptr<ov::Model> model) {
        std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
        ov::AnyMap config;
        return core->query_model(model, target_device, config);
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    ov::AnyMap configuration;
};

const std::vector<ov::AnyMap> configs = {{}};

using QueryNetworkTestSuite1NPU = QueryNetworkTestNPU;

// Test query with a supported OpenVINO model
TEST_P(QueryNetworkTestSuite1NPU, TestQueryNetworkSupported) {
    const auto supportedModel = ov::test::utils::make_conv_pool_relu();
    ov::SupportedOpsMap result;
    EXPECT_NO_THROW(result = testQueryNetwork(supportedModel));
    std::unordered_set<std::string> expected, actual;
    for (auto& op : supportedModel->get_ops()) {
        expected.insert(op->get_friendly_name());
    }
    for (auto& name : result) {
        actual.insert(name.first);
    }
    EXPECT_EQ(expected, actual);
}

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         QueryNetworkTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         QueryNetworkTestNPU::getTestCaseName);

using QueryNetworkTestSuite2NPU = QueryNetworkTestNPU;

// Test query with an unsupported ngraph function
// Tracking E#103192
TEST_P(QueryNetworkTestSuite2NPU, DISABLED_TestQueryNetworkUnsupported) {
    ov::SupportedOpsMap result;
    const auto unsupportedModel = ov::test::utils::make_conv_pool_relu_non_zero();
    EXPECT_NO_THROW(result = testQueryNetwork(unsupportedModel));
    std::unordered_set<std::string> expected, actual;
    for (auto& op : unsupportedModel->get_ops()) {
        expected.insert(op->get_friendly_name());
    }
    for (auto& name : result) {
        actual.insert(name.first);
    }
    if (actual.empty()) {
        GTEST_SKIP() << "Skip the tests since QueryNetwork is unsupported with current driver";
    } else {
        EXPECT_NE(expected, actual);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         QueryNetworkTestSuite2NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         QueryNetworkTestNPU::getTestCaseName);

using QueryNetworkTestSuite3NPU = QueryNetworkTestNPU;

// Test query with model containing an unknown Op
// Should throw error when calling testQueryNetwork, because it would fail at read_model in vcl
// TODO: E#86084, test would crash instead of throw error in Linux, need to fix that. Thus skip it for now.
TEST_P(QueryNetworkTestSuite3NPU, TestQueryNetworkThrow) {
    ov::SupportedOpsMap result;
    const auto unsupportedModel = createModelWithUnknownNode();
    EXPECT_ANY_THROW(result = testQueryNetwork(unsupportedModel));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
