// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class RoiTensorsTestsRun : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (const auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(target_device, configuration) = this->GetParam();
        OVPluginTestBase::SetUp();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

    std::shared_ptr<ov::Model> createModel(element::Type type, const PartialShape& shape, const ov::Layout& layout) {
        ResultVector res;
        ParameterVector params;

        auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
        data1->set_friendly_name("input");
        data1->get_output_tensor(0).set_names({"tensor_input"});
        data1->set_layout(layout);
        auto constant = opset8::Constant::create(type, {1}, {1});
        auto op1 = std::make_shared<ov::op::v1::Add>(data1, constant);
        op1->set_friendly_name("Add");
        auto res1 = std::make_shared<ov::op::v0::Result>(op1);
        res1->set_friendly_name("Result");
        res1->get_output_tensor(0).set_names({"tensor_output"});
        params.push_back(data1);
        res.push_back(res1);

        return std::make_shared<Model>(res, params);
    }
};

TEST_P(RoiTensorsTestsRun, CompileAndRunRoiTensorsPropertyEnabled) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");

    configuration[ov::intel_npu::inputs_with_dynamic_strides.name()] = {0};
    configuration[ov::intel_npu::outputs_with_dynamic_strides.name()] = {0};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(RoiTensorsTestsRun, CreateRoiTensorFromHostTensorAndRunInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");

    auto zero_context = core->get_default_context(target_device);
    auto input_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto input_strides = input_host_tensor.get_strides();
    auto output_strides = output_host_tensor.get_strides();

    configuration[ov::intel_npu::inputs_with_dynamic_strides.name()] = {0};
    configuration[ov::intel_npu::outputs_with_dynamic_strides.name()] = {0};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_view_tensor = ov::Tensor(ov::element::f32, shape, input_host_tensor.data(), input_strides);
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_view_tensor));

    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_view_tensor));

    OV_ASSERT_NO_THROW(req.infer());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
