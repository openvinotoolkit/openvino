// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_context.hpp"
#include "zero_backend.hpp"
#include "zero_tensor.hpp"

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

TEST_P(RoiTensorsTestsRun, CompileAndRunStridedTensorsPropertyEnabled) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input", "Result"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(RoiTensorsTestsRun, CreateStridedTensorFromHostTensorAndRunInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");
    auto zero_context = core->get_default_context(target_device);
    auto input_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto input_strides = input_host_tensor.get_strides();
    auto output_strides = output_host_tensor.get_strides();

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input", "Result"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_view_tensor = ov::Tensor(ov::element::f32, shape, input_host_tensor.data(), input_strides);
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_view_tensor));

    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_view_tensor));

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

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input", "Result"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_host_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_host_tensor, {0, 15, 15, 15}, {1, 17, 17, 17});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(RoiTensorsTestsRun, FallbackOnMemcpy) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");

    auto input_tensor = ov::Tensor{ov::element::f32, Shape{1, 10, 10, 10}};
    auto output_tensor = ov::Tensor{ov::element::f32, Shape{3, 8, 8, 8}};

    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < input_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    auto* output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input", "Result"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_tensor, {2, 4, 5, 6}, {3, 6, 7, 8});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }
}

TEST_P(RoiTensorsTestsRun, FallbackOnMemcpyRemoteTensorFromAnotherContext) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);
    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModel(element::f32, shape, "N...");

    auto input_remote_tensor = std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context,
                                                                               init_struct,
                                                                               ov::element::f32,
                                                                               Shape{1, 10, 10, 10});

    auto output_remote_tensor =
        std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context, init_struct, ov::element::f32, Shape{3, 8, 8, 8});

    auto input_tensor = make_tensor(input_remote_tensor);
    auto output_tensor = make_tensor(output_remote_tensor);

    auto input_data = static_cast<float*>(input_remote_tensor->get_original_memory());
    for (size_t i = 0; i < input_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    auto output_data = static_cast<float*>(output_remote_tensor->get_original_memory());
    for (size_t i = 0; i < output_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input", "Result"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_tensor, {2, 4, 5, 6}, {3, 6, 7, 8});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }

    init_struct = nullptr;
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
