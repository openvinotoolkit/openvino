// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class DriverCompilerAdapterDowngradeInterpolate11TestNPU :
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
               << "Interpolate11Model"
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
        ov_model = createInterpolate11Model();
        OVPluginTestBase::SetUp();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

private:
    std::shared_ptr<ov::Model> createInterpolate11Model() {
        using InterpolateAttrs = op::v11::Interpolate::InterpolateAttrs;
        using InterpolateMode = op::v11::Interpolate::InterpolateMode;
        using ShapeCalcMode = op::v11::Interpolate::ShapeCalcMode;
        using TransformMode = op::v11::Interpolate::CoordinateTransformMode;
        using NearestMode = op::v11::Interpolate::NearestMode;
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 30, 60});
        const auto scales =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, std::vector<float>{0.5f, 0.5f});
        const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
        // Only modes of NEAREST, LINEAR, LINEAR_ONNX and CUBIC are supported for ConvertInterpolate11ToInterpolate4,
        // here we use mode of NEAREST .
        const InterpolateAttrs attrs{InterpolateMode::NEAREST,
                                     ShapeCalcMode::SCALES,
                                     std::vector<size_t>{0, 0, 0, 0},
                                     std::vector<size_t>{0, 0, 0, 0},
                                     TransformMode::HALF_PIXEL,
                                     NearestMode::ROUND_PREFER_FLOOR,
                                     false,
                                     -0.75};
        const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(data, scales, axes, attrs);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(interpolate)};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{{data}}, "Interpolate-11");
    }
};

TEST_P(DriverCompilerAdapterDowngradeInterpolate11TestNPU, CheckOpsetVersion) {
    EXPECT_NO_THROW(auto compiledModel = core->compile_model(ov_model, target_device, configuration););
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
