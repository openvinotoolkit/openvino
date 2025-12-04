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
#include "openvino/runtime/iremote_tensor.hpp"
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

    std::shared_ptr<Model> createModelWithNInputs(element::Type type,
                                                  const PartialShape& shape,
                                                  const ov::Layout& layout,
                                                  size_t n = 1) {
        ResultVector res;
        ParameterVector params;

        for (size_t i = 0; i < n; i++) {
            auto index_str = std::to_string(i);
            auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
            data1->set_friendly_name("input" + index_str);
            data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
            data1->set_layout(layout);
            auto constant = opset8::Constant::create(type, {1}, {1});
            auto op1 = std::make_shared<ov::op::v1::Add>(data1, constant);
            op1->set_friendly_name("Add" + index_str);
            auto res1 = std::make_shared<ov::op::v0::Result>(op1);
            res1->set_friendly_name("Result" + index_str);
            res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
            params.push_back(data1);
            res.push_back(res1);
        }

        return std::make_shared<Model>(res, params);
    }
};

TEST_P(RoiTensorsTestsRun, CompileAndRunStridedTensorsPropertyEnabled) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0", "dummy"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(RoiTensorsTestsRun, CompileAndRunStridedTensorsPropertyEnabledInternalOperator) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");
    auto zero_context = core->get_default_context(target_device);
    auto input_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto input_strides = input_host_tensor.get_strides();
    auto output_strides = output_host_tensor.get_strides();

    configuration[ov::intel_npu::enable_strides_for.name()] = "input0,Result0,dummy";

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_view_tensor = ov::Tensor(ov::element::f32, shape, input_host_tensor.data(), input_strides);
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_view_tensor));

    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_view_tensor));

    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(RoiTensorsTestsRun, CreateStridedTensorFromHostTensorAndRunInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");
    auto zero_context = core->get_default_context(target_device);
    auto input_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto input_strides = input_host_tensor.get_strides();
    auto output_strides = output_host_tensor.get_strides();

    auto* input_data = input_host_tensor.data<float>();
    for (size_t i = 0; i < input_host_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    auto* output_data = output_host_tensor.data<float>();
    for (size_t i = 0; i < output_host_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_view_tensor = ov::Tensor(ov::element::f32, shape, input_host_tensor.data(), input_strides);
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_view_tensor));

    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_view_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_view_tensor = ov::Tensor(ov::element::f32, shape);
    output_view_tensor.copy_to(check_out_view_tensor);
    auto* check_data = check_out_view_tensor.data<float>();
    for (size_t i = 0; i < check_out_view_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }
}

TEST_P(RoiTensorsTestsRun, SetStridedTensorForUnexpectedTensorExpectedThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");
    auto zero_context = core->get_default_context(target_device);
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto output_strides = output_host_tensor.get_strides();

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0"};
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    EXPECT_THROW(req.set_output_tensor(output_view_tensor), ov::Exception);
}

TEST_P(RoiTensorsTestsRun, SetStridedMultipleOutputTensorForUnexpectedTensorExpectedThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...", 2);
    auto zero_context = core->get_default_context(target_device);
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});
    auto output_strides = output_host_tensor.get_strides();

    configuration[ov::intel_npu::enable_strides_for.name()] =
        std::vector<std::string>{"input0", "input1", "tensor_output0"};
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
    ov::Tensor output_view_tensor = ov::Tensor(ov::element::f32, shape, output_host_tensor.data(), output_strides);
    EXPECT_THROW(req.set_output_tensor(1, output_view_tensor), ov::Exception);
}

TEST_P(RoiTensorsTestsRun, CreateRoiTensorFromHostTensorAndRunInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

    auto zero_context = core->get_default_context(target_device);
    auto input_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});

    auto* input_data = input_host_tensor.data<float>();
    for (size_t i = 0; i < input_host_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    auto* output_data = output_host_tensor.data<float>();
    for (size_t i = 0; i < output_host_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_host_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_host_tensor, {0, 15, 15, 15}, {1, 17, 17, 17});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }
}

TEST_P(RoiTensorsTestsRun, FallbackOnMemcpy) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

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

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

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

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);
    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

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

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

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

TEST_P(RoiTensorsTestsRun, FallbackOnMemcpyRemoteTensorFromAnotherContextCopyToAnotherRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);
    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

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

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_tensor, {2, 4, 5, 6}, {3, 6, 7, 8});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context,
                                                                                init_struct,
                                                                                ov::element::f32,
                                                                                Shape{1, 8, 8, 10});
    auto check_out_roi_ov_tensor = make_tensor(check_out_roi_tensor);
    ov::Tensor check_out_roi_tensor_roi = ov::Tensor(check_out_roi_ov_tensor, {0, 4, 5, 6}, {1, 6, 7, 8});
    output_roi_tensor.copy_to(check_out_roi_tensor_roi);

    auto check_out_roi_roi_tensor = ov::Tensor(ov::element::f32, shape);
    check_out_roi_tensor_roi.copy_to(check_out_roi_roi_tensor);
    auto* check_data = check_out_roi_roi_tensor.data<float>();
    for (size_t i = 0; i < req.get_output_tensor().get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }

    init_struct = nullptr;
}

TEST_P(RoiTensorsTestsRun, FallbackOnMemcpyRemoteTensorFromAnotherContextCopyFromAnotherRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);
    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

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

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_tensor, {2, 4, 5, 6}, {3, 6, 7, 8});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context,
                                                                                init_struct,
                                                                                ov::element::f32,
                                                                                Shape{1, 8, 8, 10});
    auto check_out_roi_ov_tensor = make_tensor(check_out_roi_tensor);
    ov::Tensor check_out_roi_tensor_roi = ov::Tensor(check_out_roi_ov_tensor, {0, 4, 5, 6}, {1, 6, 7, 8});
    auto check_out_roi_remote_tensor_roi =
        std::dynamic_pointer_cast<ov::IRemoteTensor>(get_tensor_impl(check_out_roi_tensor_roi)._ptr);
    check_out_roi_remote_tensor_roi->copy_from(get_tensor_impl(output_roi_tensor)._ptr);

    auto check_out_roi_roi_tensor = ov::Tensor(ov::element::f32, shape);
    check_out_roi_remote_tensor_roi->copy_to(get_tensor_impl(check_out_roi_roi_tensor)._ptr);
    auto* check_data = check_out_roi_roi_tensor.data<float>();
    for (size_t i = 0; i < req.get_output_tensor().get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }

    init_struct = nullptr;
}

TEST_P(RoiTensorsTestsRun, ImportStandardAllocation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 1024};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

    auto shape_input = Shape{1, 4, 8, 1024};
    auto shape_output = Shape{1, 2, 4, 2048};

    auto input_data = static_cast<float*>(
        ::operator new(ov::shape_size(shape_input) * ov::element::f32.size(), std::align_val_t(4096)));
    auto output_data = static_cast<float*>(
        ::operator new(ov::shape_size(shape_output) * ov::element::f32.size(), std::align_val_t(4096)));

    auto input_tensor = ov::Tensor{ov::element::f32, shape_input, input_data};
    auto output_tensor = ov::Tensor{ov::element::f32, shape_output, output_data};

    for (size_t i = 0; i < input_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    for (size_t i = 0; i < output_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_tensor, {0, 0, 0, 0}, {1, 2, 2, 1024});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_tensor, {0, 0, 2, 0}, {1, 2, 4, 1024});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }

    req = {};
    check_out_roi_tensor = {};
    input_roi_tensor = {};
    output_roi_tensor = {};
    input_tensor = {};
    output_tensor = {};

    ::operator delete(input_data, std::align_val_t(4096));
    ::operator delete(output_data, std::align_val_t(4096));
}

TEST_P(RoiTensorsTestsRun, RunWithRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto zero_context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto input_remote_tensor = zero_context.create_l0_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto output_remote_tensor = zero_context.create_l0_host_tensor(ov::element::f32, Shape{3, 8, 8, 8});

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...");

    auto input_data = static_cast<float*>(input_remote_tensor.get());
    for (size_t i = 0; i < input_remote_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    auto output_data = static_cast<float*>(output_remote_tensor.get());
    for (size_t i = 0; i < output_remote_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input_roi_tensor = ov::Tensor(input_remote_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(input_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_remote_tensor, {2, 5, 6, 6}, {3, 7, 8, 8});
    OV_ASSERT_NO_THROW(req.set_output_tensor(output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);

    std::cout << "check_out_roi_tensor.data(): " << check_out_roi_tensor.data() << std::endl;
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }
}

TEST_P(RoiTensorsTestsRun, MultipleIOCreateRoiTensorFromHostTensorAndRunInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto supportedProperties =
        core->get_property(target_device, supported_properties.name()).as<std::vector<PropertyName>>();
    bool isStridedEnabled =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == ov::intel_npu::enable_strides_for.name();
        });

    if (!isStridedEnabled) {
        GTEST_SKIP() << "NPU_ENABLE_STRIDES_FOR property is not supported";
    }

    auto shape = Shape{1, 2, 2, 2};
    ov::CompiledModel compiled_model;
    auto model = createModelWithNInputs(element::f32, shape, "N...", 2);

    auto zero_context = core->get_default_context(target_device);
    auto input0_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 10, 10, 10});
    auto input1_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{2, 12, 15, 8});
    auto output_host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{1, 25, 25, 25});

    auto* input_data = input0_host_tensor.data<float>();
    for (size_t i = 0; i < input0_host_tensor.get_size(); ++i) {
        input_data[i] = 50.0f;
    }

    input_data = input1_host_tensor.data<float>();
    for (size_t i = 0; i < input0_host_tensor.get_size(); ++i) {
        input_data[i] = 75.0f;
    }

    auto* output_data = output_host_tensor.data<float>();
    for (size_t i = 0; i < output_host_tensor.get_size(); ++i) {
        output_data[i] = 10.0f;
    }

    configuration[ov::intel_npu::enable_strides_for.name()] = std::vector<std::string>{"input0", "input1", "Result0"};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());

    ov::Tensor input0_roi_tensor = ov::Tensor(input0_host_tensor, {0, 4, 4, 4}, {1, 6, 6, 6});
    OV_ASSERT_NO_THROW(req.set_input_tensor(0, input0_roi_tensor));

    ov::Tensor input1_roi_tensor = ov::Tensor(input1_host_tensor, {1, 5, 7, 6}, {2, 7, 9, 8});
    OV_ASSERT_NO_THROW(req.set_input_tensor(1, input1_roi_tensor));

    ov::Tensor output_roi_tensor = ov::Tensor(output_host_tensor, {0, 15, 15, 15}, {1, 17, 17, 17});
    OV_ASSERT_NO_THROW(req.set_output_tensor(0, output_roi_tensor));

    OV_ASSERT_NO_THROW(req.infer());

    auto check_out_roi_tensor = ov::Tensor(ov::element::f32, shape);
    output_roi_tensor.copy_to(check_out_roi_tensor);
    auto* check_data = check_out_roi_tensor.data<float>();
    for (size_t i = 0; i < check_out_roi_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 51.0f);
    }

    auto check_out1_tensor = req.get_output_tensor(1);
    check_data = check_out1_tensor.data<float>();
    for (size_t i = 0; i < check_out1_tensor.get_size(); ++i) {
        EXPECT_EQ(check_data[i], 76.0f);
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
