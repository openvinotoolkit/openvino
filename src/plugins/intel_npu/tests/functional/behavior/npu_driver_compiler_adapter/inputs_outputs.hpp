// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"

#include "openvino/op/relu.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {

class DriverCompilerAdapterInputsOutputsTestNPU :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "model="
               << "SimpleReluModel"
               << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        ov_model = createSimpleReluModel();
        OVPluginTestBase::SetUp();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

private:
    std::shared_ptr<ov::Model> createSimpleReluModel() {
        auto arg0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
        arg0->set_friendly_name("data");
        arg0->get_output_tensor(0).set_names({"input"});

        auto relu = std::make_shared<ov::op::v0::Relu>(arg0);
        relu->set_friendly_name("relu");
        relu->get_output_tensor(0).set_names({"relu_res"});
        auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
        f->validate_nodes_and_infer_types();
        return f;
    }
};

TEST_P(DriverCompilerAdapterInputsOutputsTestNPU, CheckInOutputs) {
    auto compiledModel = core->compile_model(ov_model, target_device, configuration);
    for (auto&& input : compiledModel.inputs()) {
        std::string in_name;
        std::string in_node_name;
        try {
            for (const auto& name : input.get_names()) {
                in_name += name + " , ";
            }
            in_name = in_name.substr(0, in_name.size() - 3);
        } catch (const ov::Exception&) {
        }

        try {
            in_node_name = input.get_node()->get_friendly_name();
        } catch (const ov::Exception&) {
        }
        EXPECT_EQ(in_name, "input");
        EXPECT_EQ(in_node_name, "data");
    }

    for (auto&& output : compiledModel.outputs()) {
        std::string out_name;
        std::string out_node_name;

        try {
            for (const auto& name : output.get_names()) {
                out_name += name + " , ";
            }
            out_name = out_name.substr(0, out_name.size() - 3);
        } catch (const ov::Exception&) {
        }
        try {
            out_node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        } catch (const ov::Exception&) {
        }
        EXPECT_EQ(out_name, "relu_res");
        EXPECT_EQ(out_node_name, "relu");
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
