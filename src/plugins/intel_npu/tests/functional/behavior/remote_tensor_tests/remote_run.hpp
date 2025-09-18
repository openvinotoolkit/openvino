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
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/op.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "overload/overload_test_utils_npu.hpp"
#include "remote_context.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class RemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                       public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::CompiledModel compiled_model;

    std::string m_cache_dir;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

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

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = getDefaultNGraphFunctionForTheDeviceNPU();  // FIXME: E#80555
    }

    std::string generateCacheDirName(const std::string& test_name) {
        using namespace std::chrono;
        // Generate unique file names based on test name, thread id and timestamp
        // This allows execution of tests in parallel (stress mode)
        auto hash = std::to_string(std::hash<std::string>()(test_name));
        std::stringstream ss;
        auto ts = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch());
        ss << hash << "_"
           << "_" << ts.count();
        return ss.str();
    }

    void TearDown() override {
        if (!m_cache_dir.empty()) {
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::PluginCache::get().reset();
            ov::test::utils::removeFilesWithExt(m_cache_dir, "blob");
            ov::test::utils::removeDir(m_cache_dir);
        }

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

TEST_P(RemoteRunTests, CheckIsContinuousHostTensorScalar) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_context = core->get_default_context(target_device);

    auto host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{});
    auto data = host_tensor.data();
    auto strides = host_tensor.get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(RemoteRunTests, CheckIsContinuousHostTensor1Dimension) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_context = core->get_default_context(target_device);

    auto host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{128});
    auto data = host_tensor.data();
    auto strides = host_tensor.get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(RemoteRunTests, CheckIsContinuousHostTensor2Dimensions) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_context = core->get_default_context(target_device);

    auto host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{32, 128});
    auto data = host_tensor.data();
    auto strides = host_tensor.get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);
}

TEST_P(RemoteRunTests, CheckIsContinuousHostTensor3Dimensions) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_context = core->get_default_context(target_device);

    auto host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{5, 32, 128});
    auto data = host_tensor.data();
    auto strides = host_tensor.get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(RemoteRunTests, CheckIsContinuousHostTensor4Dimensions) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_context = core->get_default_context(target_device);

    auto host_tensor = zero_context.create_host_tensor(ov::element::f32, Shape{3, 5, 32, 128});
    auto data = host_tensor.data();
    auto strides = host_tensor.get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 5, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 2, 5, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{3, 5, 32, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 1, 32}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBuf) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    auto zero_context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, zero_context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto tensor = inference_request.get_input_tensor();

    auto remote_tensor =
        zero_context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);
    tensor = {};

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckRemoteTensorImportFile0) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 16, 16, 16};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    const std::string filename = "CheckRemoteTensorImportFile0.bin";
    std::vector<float> data(shape_size, 5.0f);

    {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        out.close();
    }

    ov::InferRequest inference_request;

    auto zero_context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, zero_context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto remote_tensor = zero_context.create_tensor(ov::element::f32, shape, ov::intel_npu::FileDescriptor{filename});

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_output_tensor();
    float* output_tensor_data = reinterpret_cast<float*>(output_tensor.data());

    float expected_result = 6.0f;
    for (size_t j = 0; j < output_tensor.get_size(); ++j) {
        EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
            << " Expected=" << expected_result << ", actual=" << output_tensor_data[j] << " for index " << j;
    }

    // destroy inference request and remote tensor to ensure that the file is closed
    inference_request = {};
    check_remote_tensor = {};
    remote_tensor = {};

    std::filesystem::remove(filename);
}

TEST_P(RemoteRunTests, CheckRemoteTensorImportFile1) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 16, 16, 16};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    const std::string filename = "CheckRemoteTensorImportFile1.bin";
    std::vector<float> data(shape_size, 5.0f);

    {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        out.close();
    }

    ov::InferRequest inference_request;

    auto context = core->get_default_context(target_device);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::MMAPED_FILE},
                         {ov::intel_npu::file_descriptor.name(), ov::intel_npu::FileDescriptor{filename}},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto remote_tensor = context.create_tensor(ov::element::f32, shape, params);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_output_tensor();
    float* output_tensor_data = reinterpret_cast<float*>(output_tensor.data());

    float expected_result = 6.0f;
    for (size_t j = 0; j < output_tensor.get_size(); ++j) {
        EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
            << " Expected=" << expected_result << ", actual=" << output_tensor_data[j] << " for index " << j;
    }

    // destroy inference request and remote tensor to ensure that the file is closed
    inference_request = {};
    remote_tensor = {};

    std::filesystem::remove(filename);
}

TEST_P(RemoteRunTests, CheckRemoteTensorImportFile2) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 16, 16, 16};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    const std::string filename = "CheckRemoteTensorImportFile2.bin";
    std::vector<float> data(shape_size, 5.0f);

    {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        out.close();
    }

    ov::InferRequest inference_request;

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::MMAPED_FILE},
                         {ov::intel_npu::file_descriptor.name(), ov::intel_npu::FileDescriptor{filename}},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto remote_tensor = context.create_tensor(ov::element::f32, shape);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_output_tensor();
    float* output_tensor_data = reinterpret_cast<float*>(output_tensor.data());

    float expected_result = 6.0f;
    for (size_t j = 0; j < output_tensor.get_size(); ++j) {
        EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
            << " Expected=" << expected_result << ", actual=" << output_tensor_data[j] << " for index " << j;
    }

    // destroy inference request and remote tensor to ensure that the file is closed
    inference_request = {};
    remote_tensor = {};

    std::filesystem::remove(filename);
}

TEST_P(RemoteRunTests, CheckRemoteTensorImportFile3) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 5, 5, 5};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    const std::string filename = "CheckRemoteTensorImportFile3.bin";
    std::vector<float> data(shape_size, 5.0f);

    {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        out.close();
    }

    ov::InferRequest inference_request;

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::MMAPED_FILE},
                         {ov::intel_npu::file_descriptor.name(), ov::intel_npu::FileDescriptor{filename}},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto remote_tensor = context.create_tensor(ov::element::f32, shape);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_output_tensor();
    float* output_tensor_data = reinterpret_cast<float*>(output_tensor.data());

    float expected_result = 6.0f;
    for (size_t j = 0; j < output_tensor.get_size(); ++j) {
        EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
            << " Expected=" << expected_result << ", actual=" << output_tensor_data[j] << " for index " << j;
    }

    // destroy inference request and remote tensor to ensure that the file is closed
    inference_request = {};
    remote_tensor = {};

    std::filesystem::remove(filename);
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBufSetPropertyInContext) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto tensor = inference_request.get_input_tensor();
    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape());
    tensor = {};

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckRemoteTensorSetOnlyTensorType) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    ov::AnyMap params = {{ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto tensor = inference_request.get_input_tensor();
    ASSERT_THROW(auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape()), ov::Exception);
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBufSetPropertyInContextandChangedInTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    ov::AnyMap paramsContext = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                                {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, paramsContext);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    ov::AnyMap paramsTensor = {{ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::BINDED}}};

    auto tensor = inference_request.get_input_tensor();
    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), paramsTensor);
    tensor = {};

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBufSetPropertyInContextandChangedInTensorExpectToFail) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    ov::AnyMap paramsContext = {{ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, paramsContext);
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, context, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    ov::AnyMap paramsTensor = {{ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::BINDED}}};

    auto tensor = inference_request.get_input_tensor();
    ASSERT_THROW(auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), paramsTensor),
                 ov::Exception);
}

TEST_P(RemoteRunTests, CheckImportModelPath) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    auto zero_context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    m_cache_dir = generateCacheDirName(GetTestName());
    core->set_property({ov::cache_dir(m_cache_dir)});
    auto compiled_model_no_cache = core->compile_model(ov_model, zero_context, configuration);
    compiled_model = core->compile_model(ov_model, zero_context, configuration);

    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto tensor = inference_request.get_input_tensor();

    auto remote_tensor =
        zero_context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);
    tensor = {};

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBufChangingTensors) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    // set input remote tensor
    auto tensor = inference_request.get_input_tensor();
    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto remote_tensor =
        context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);

    ov::Tensor check_remote_tensor;
    OV_ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random input tensor
    ov::Tensor random_tensor_input{ov::element::f32, tensor.get_shape()};
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random output tensor
    auto output_tensor = inference_request.get_output_tensor();
    ov::Tensor outputrandom_tensor_input{ov::element::f32, output_tensor.get_shape()};
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(outputrandom_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set output remote tensor
    auto remote_output_tensor = inference_request.get_output_tensor();
    auto output_remote_tensor = context.create_tensor(ov::element::f32, remote_output_tensor.get_shape());
    remote_output_tensor = {};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRuns) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;

    {
        auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();

        auto remote_tensor =
            context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);
        memset(remote_tensor.get(), 1, tensor.get_byte_size());
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        first_output = inference_request.get_output_tensor(0);
    }

    compiled_model = {};
    inference_request = {};

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        float* data = new float[tensor.get_byte_size() / sizeof(float)];
        memset(data, 1, tensor.get_byte_size());
        ov::Tensor input_data_tensor{ov::element::f32, tensor.get_shape(), data};
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        second_output = inference_request.get_output_tensor(0);

        delete[] data;
    }

    EXPECT_NE(first_output.data(), second_output.data());
    EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromRemoteTensorFromDifferentContext) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;

    {
        auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();

        ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                             {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

        auto init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();
        std::shared_ptr<::intel_npu::IEngineBackend> engine_backend =
            std::make_shared<::intel_npu::ZeroEngineBackend>();
        auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
        auto remote_tensor_impl = std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context,
                                                                                  init_struct,
                                                                                  ov::element::f32,
                                                                                  tensor.get_shape());
        ov::Tensor remote_tensor = make_tensor(remote_tensor_impl);

        memset(remote_tensor_impl->get_original_memory(), 1, remote_tensor_impl->get_byte_size());
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        first_output = inference_request.get_output_tensor(0);
    }

    compiled_model = {};
    inference_request = {};

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        float* data = new float[tensor.get_byte_size() / sizeof(float)];
        memset(data, 1, tensor.get_byte_size());
        ov::Tensor input_data_tensor{ov::element::f32, tensor.get_shape(), data};
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        second_output = inference_request.get_output_tensor(0);

        delete[] data;
    }

    EXPECT_NE(first_output.data(), second_output.data());
    EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    void* first_output = nullptr;
    ov::intel_npu::level_zero::ZeroBufferTensor remote_output_tensor;
    ov::Tensor second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto input_tensor = inference_request.get_input_tensor();
        auto output_tensor = inference_request.get_output_tensor();
        const auto byte_size = input_tensor.get_byte_size();
        auto input_shape = input_tensor.get_shape();
        auto output_shape = output_tensor.get_shape();
        input_tensor = {};
        output_tensor = {};

        auto remote_input_tensor =
            context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
        remote_output_tensor = context.create_l0_host_tensor(ov::element::f32, output_shape);

        memset(remote_input_tensor.get(), 99, byte_size);
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
        OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());

        first_output = remote_output_tensor.get();
    }

    compiled_model = {};
    inference_request = {};

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        memset(tensor.data(), 99, tensor.get_byte_size());
        OV_ASSERT_NO_THROW(inference_request.infer());

        second_output = inference_request.get_output_tensor(0);
    }

    EXPECT_NE(first_output, second_output.data());
    EXPECT_EQ(memcmp(first_output, second_output.data(), second_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors2) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    void* first_output = NULL;
    void* second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor1 =
        context.create_tensor(ov::element::f32, output_shape).as<ov::intel_npu::level_zero::ZeroBufferTensor>();

    memset(remote_input_tensor.get(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor1));
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = remote_output_tensor1.get();

    float* data = new float[byte_size / sizeof(float)];
    memset(data, 99, byte_size);
    ov::Tensor input_data_tensor{ov::element::f32, input_shape, data};
    auto remote_output_tensor2 =
        context.create_tensor(ov::element::f32, output_shape).as<ov::intel_npu::level_zero::ZeroBufferTensor>();

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor2));
    OV_ASSERT_NO_THROW(inference_request.infer());
    second_output = remote_output_tensor2.get();

    EXPECT_NE(first_output, second_output);
    EXPECT_EQ(memcmp(first_output, second_output, remote_output_tensor2.get_byte_size()), 0);

    delete[] data;
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors3) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    void* second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();
    memset(tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = inference_request.get_output_tensor(0);

    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor = context.create_l0_host_tensor(ov::element::f32, output_shape);

    memset(remote_input_tensor.get(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
    second_output = remote_output_tensor.get();

    EXPECT_NE(first_output.data(), second_output);
    EXPECT_EQ(memcmp(first_output.data(), second_output, first_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensorsHostTensor1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();
    memset(tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = inference_request.get_output_tensor();

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, tensor.get_shape());
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, first_output.get_shape());

    memset(l0_host_input_tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(l0_host_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(l0_host_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(first_output.data(), l0_host_output_tensor.data());
    EXPECT_EQ(memcmp(first_output.data(), l0_host_output_tensor.data(), first_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensorsHostTensor2) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor =
        context.create_l0_host_tensor(ov::element::f32, output_shape, ov::intel_npu::TensorType::OUTPUT);
    memset(remote_input_tensor.get(), 1, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, input_shape);
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, output_shape);

    memset(l0_host_input_tensor.data(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(l0_host_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(l0_host_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(remote_output_tensor.get(), l0_host_output_tensor.data());
    EXPECT_NE(memcmp(remote_output_tensor.get(), l0_host_output_tensor.data(), remote_output_tensor.get_byte_size()),
              0);
}

TEST_P(RemoteRunTests, checkResultsAfterChangingStateTensors) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    testing::internal::Random random(1);
    ov::Tensor input_tensor;

    auto original_shape = Shape{1, 10, 10, 10};
    auto shape_size = ov::shape_size(original_shape);
    auto model = createModelWithStates(element::f32, original_shape);

    auto context = core->get_default_context(target_device);

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    auto input = compiled_model.input();
    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = static_cast<float>(random.Generate(10));
    }

    auto states = inference_request.query_state();

    auto tensor_state = states[0].get_state();
    auto tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor0 = context.create_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor1 = context.create_host_tensor(ov::element::f32, tensor_state_shape);

    states[0].set_state(l0_host_tensor0);
    states[0].reset();
    states[1].set_state(l0_host_tensor1);
    states[1].reset();

    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_tensor("sigmod_state");
    auto output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }

    auto tensor_size = l0_host_tensor0.get_size();
    auto state_data = static_cast<float*>(l0_host_tensor0.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor1.get_size();
    state_data = static_cast<float*>(l0_host_tensor1.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_state = states[0].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor2 = context.create_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor3 = context.create_host_tensor(ov::element::f32, tensor_state_shape);

    states[0].set_state(l0_host_tensor2);
    states[1].set_state(l0_host_tensor3);

    tensor_size = l0_host_tensor2.get_size();
    state_data = static_cast<float*>(l0_host_tensor2.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    tensor_size = l0_host_tensor3.get_size();
    state_data = static_cast<float*>(l0_host_tensor3.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    OV_ASSERT_NO_THROW(inference_request.infer());

    tensor_size = l0_host_tensor2.get_size();
    state_data = static_cast<float*>(l0_host_tensor2.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor3.get_size();
    state_data = static_cast<float*>(l0_host_tensor3.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }
}

TEST_P(RemoteRunTests, checkResultsAfterChangingStateTensorsWithRemoteTensors) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    testing::internal::Random random(1);
    ov::Tensor input_tensor;

    auto original_shape = Shape{1, 2, 2, 2};
    auto shape_size = ov::shape_size(original_shape);
    auto model = createModelWithStates(element::f32, original_shape);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    auto input = compiled_model.input();
    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = static_cast<float>(random.Generate(10));
    }

    auto states = inference_request.query_state();

    auto tensor_state = states[0].get_state();
    auto tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor0 = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor1 = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    states[0].set_state(l0_host_tensor0);
    states[0].reset();
    states[1].set_state(l0_host_tensor1);
    states[1].reset();

    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_tensor("sigmod_state");
    auto output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }

    auto tensor_size = l0_host_tensor0.get_size();
    auto state_data = static_cast<float*>(l0_host_tensor0.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor1.get_size();
    state_data = static_cast<float*>(l0_host_tensor1.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_state = states[0].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor2 = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor3 = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    states[0].set_state(l0_host_tensor2);
    states[1].set_state(l0_host_tensor3);

    tensor_size = l0_host_tensor2.get_size();
    state_data = static_cast<float*>(l0_host_tensor2.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    tensor_size = l0_host_tensor3.get_size();
    state_data = static_cast<float*>(l0_host_tensor3.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    OV_ASSERT_NO_THROW(inference_request.infer());

    tensor_size = l0_host_tensor2.get_size();
    state_data = static_cast<float*>(l0_host_tensor2.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor3.get_size();
    state_data = static_cast<float*>(l0_host_tensor3.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }
}

TEST_P(RemoteRunTests, checkResultsAfterChangingStateDataWithRemoteAndRandomTensors0) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    testing::internal::Random random(1);
    ov::Tensor input_tensor;

    auto original_shape = Shape{1, 10, 10, 10};
    auto shape_size = ov::shape_size(original_shape);
    auto model = createModelWithStates(element::f32, original_shape);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    auto input = compiled_model.input();
    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = static_cast<float>(random.Generate(10));
    }

    auto states = inference_request.query_state();

    auto tensor_state = states[0].get_state();
    auto tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto byte_size = tensor_state.get_byte_size();
    float* data = new float[byte_size / sizeof(float)];
    ov::Tensor random_tensor{ov::element::f32, tensor_state_shape, data};

    states[0].set_state(l0_host_tensor);
    states[0].reset();
    states[1].set_state(random_tensor);
    states[1].reset();

    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_tensor("sigmod_state");
    auto output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }

    auto tensor_size = l0_host_tensor.get_size();
    auto state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor.get_size();
    state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    OV_ASSERT_NO_THROW(inference_request.infer());

    tensor_size = l0_host_tensor.get_size();
    state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }
}

TEST_P(RemoteRunTests, checkResultsAfterChangingStateDataWithRemoteAndRandomTensors1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    testing::internal::Random random(1);
    ov::Tensor input_tensor;

    auto original_shape = Shape{1, 10, 10, 10};
    auto shape_size = ov::shape_size(original_shape);
    auto model = createModelWithStates(element::f32, original_shape);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    auto input = compiled_model.input();
    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = static_cast<float>(random.Generate(10));
    }

    auto states = inference_request.query_state();

    auto tensor_state = states[0].get_state();
    auto tensor_state_shape = tensor_state.get_shape();
    auto l0_host_tensor = context.create_l0_host_tensor(ov::element::f32, tensor_state_shape);

    tensor_state = states[1].get_state();
    tensor_state_shape = tensor_state.get_shape();
    auto byte_size = tensor_state.get_byte_size();
    float* data = new float[byte_size / sizeof(float)];
    ov::Tensor random_tensor{ov::element::f32, tensor_state_shape, data};

    auto tensor_size = l0_host_tensor.get_size();
    auto state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        state_data[i] = 1.0f;
    }

    states[0].set_state(l0_host_tensor);
    states[1].set_state(random_tensor);

    OV_ASSERT_NO_THROW(inference_request.infer());

    tensor_size = l0_host_tensor.get_size();
    state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(input_data[i], state_data[i], 1e-5);
    }

    states[0].reset();
    states[1].reset();

    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_tensor("sigmod_state");
    auto output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }

    tensor_size = l0_host_tensor.get_size();
    state_data = static_cast<float*>(l0_host_tensor.get());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }

    tensor_size = random_tensor.get_size();
    state_data = static_cast<float*>(random_tensor.data());
    for (size_t i = 0; i < tensor_size; ++i) {
        EXPECT_NEAR(0.0, state_data[i], 1e-5);
    }
}

TEST_P(RemoteRunTests, CheckContextFromDifferentOvCores) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core core0;
    ov::Core core1;

    auto context0 = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto context1 = core0.get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto context2 = core1.get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    EXPECT_EQ(context0.get(), context1.get());
    EXPECT_EQ(context1.get(), context2.get());
}

TEST_P(RemoteRunTests, CheckContextFromDifferentDestroyedOvCores) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::RemoteContext context1, context2;
    auto context0 = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    {
        ov::Core internal_core;
        context1 = internal_core.get_default_context(target_device);
    }

    {
        ov::Core internal_core;
        context2 = internal_core.get_default_context(target_device);
    }

    EXPECT_EQ(context0.get(), context1.get_params().at(ov::intel_npu::l0_context.name()).as<void*>());
    EXPECT_EQ(context1.get_params().at(ov::intel_npu::l0_context.name()).as<void*>(),
              context2.get_params().at(ov::intel_npu::l0_context.name()).as<void*>());
}

TEST_P(RemoteRunTests, SetMultipleDifferentTensors) {
    auto shape = Shape{1, 16, 16, 16};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    compiled_model = core->compile_model(model, target_device, configuration);

    const int inferences = 32;
    ov::InferRequest inference_request;
    ov::Tensor input_tensor;
    std::array<ov::Tensor, inferences> output_tensor;

    input_tensor = ov::Tensor{ov::element::f32, shape};
    auto* input_tensor_data = reinterpret_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape_size; ++i) {
        input_tensor_data[i] = 0.f;
    }

    inference_request = compiled_model.create_infer_request();
    for (int i = 0; i < inferences; i++) {
        auto tensor = inference_request.get_output_tensor(0);

        if (i % 5 == 0) {
            output_tensor[i] = ov::Tensor{ov::element::f32, tensor.get_shape()};
        } else if (i % 5 == 1) {
            output_tensor[i] = context.create_l0_host_tensor(ov::element::f32, tensor.get_shape());
        } else if (i % 5 == 2) {
            output_tensor[i] = ov::Tensor{ov::element::f32, tensor.get_shape()};
        } else if (i % 5 == 3) {
            output_tensor[i] = context.create_host_tensor(ov::element::f32, tensor.get_shape());
        } else if (i % 5 == 4) {
            output_tensor[i] = ov::Tensor{ov::element::f32, tensor.get_shape()};
        }
    }

    inference_request.set_input_tensor(input_tensor);
    inference_request.set_output_tensor(output_tensor[0]);
    inference_request.infer();  // Adds '1' to each element

    for (int i = 1; i < inferences; i++) {
        inference_request.set_output_tensor(output_tensor[i]);
        inference_request.set_input_tensor(output_tensor[i - 1]);
        inference_request.infer();  // Adds '1' to each element
    }

    float expected_result = 1.f;

    for (int i = 0; i < inferences; i++) {
        float* output_tensor_data;
        if (i % 5 == 1) {
            auto remote_tensor = output_tensor[i].as<ov::intel_npu::level_zero::ZeroBufferTensor>();
            output_tensor_data = static_cast<float*>(remote_tensor.get());
        } else {
            output_tensor_data = reinterpret_cast<float*>(output_tensor[i].data());
        }
        for (size_t j = 0; j < shape_size; ++j) {
            EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
                << "Output=" << i << " Expected=" << expected_result << ", actual=" << output_tensor_data[j]
                << " for index " << j;
        }
        expected_result++;
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
