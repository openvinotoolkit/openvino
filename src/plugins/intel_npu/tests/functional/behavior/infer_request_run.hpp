// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <exception>
#include <random>
#include <thread>

#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_infer_request/inference.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "overload/overload_test_utils_npu.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class InferRequestRunTests : public ov::test::behavior::OVPluginTestBase,
                             public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::CompiledModel compiled_model;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
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

TEST_P(InferRequestRunTests, AllocatorCanDisposeBlobWhenOnlyInferRequestIsInScope) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        ov::InferRequest req;
        ov::Tensor outputTensor;
        {
            core.reset();
            ov::test::utils::PluginCache::get().reset();
        }
    }
    std::cout << "Plugin should be unloaded from memory at this point" << std::endl;
}

TEST_P(InferRequestRunTests, MultipleExecutorStreamsTestsSyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiled_model.input());
    OV_ASSERT_NO_THROW(output = compiled_model.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    std::array<ov::InferRequest, inferReqNumber> inferReqs;
    std::array<std::thread, inferReqNumber> inferReqsThreads;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i] = compiled_model.create_infer_request());
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(input));
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        ov::InferRequest& infReq = inferReqs[i];
        inferReqsThreads[i] = std::thread([&infReq]() -> void {
            OV_ASSERT_NO_THROW(infReq.infer());
        });
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqsThreads[i].join();
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(output));
    }
}

TEST_P(InferRequestRunTests, MultipleExecutorStreamsTestsAsyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiled_model.input());
    OV_ASSERT_NO_THROW(output = compiled_model.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    std::array<ov::InferRequest, inferReqNumber> inferReqs;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i] = compiled_model.create_infer_request());
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(input));
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i].start_async());
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqs[i].wait();
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(output));
    }
}

TEST_P(InferRequestRunTests, MultipleExecutorTestsSyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiled_model.input());
    OV_ASSERT_NO_THROW(output = compiled_model.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    ov::InferRequest inferReq;
    ov::Tensor input_tensor;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReq = compiled_model.create_infer_request());
        OV_ASSERT_NO_THROW(input_tensor = inferReq.get_tensor(input));
        OV_ASSERT_NO_THROW(inferReq.set_input_tensor(input_tensor));
        OV_ASSERT_NO_THROW(inferReq.infer());
        OV_ASSERT_NO_THROW(inferReq.get_tensor(output));
    }
}

TEST_P(InferRequestRunTests, CheckOutputDataFromTwoRuns) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;
    float* data;

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        const size_t byte_size = tensor.get_byte_size();
        data = new float[byte_size / sizeof(float)];
        memset(data, 1, byte_size);
        ov::Tensor input_data_tensor{ov::element::f32, tensor.get_shape(), data};
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
    }
    first_output = inference_request.get_output_tensor(0);

    for (int i = 0; i < 10; i++) {
        delete[] data;
        compiled_model = {};
        inference_request = {};

        {
            OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
            OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
            auto tensor = inference_request.get_input_tensor();
            const size_t byte_size = tensor.get_byte_size();
            data = new float[byte_size / sizeof(float)];
            memset(data, 1, byte_size);
            ov::Tensor input_data_tensor{ov::element::f32, tensor.get_shape(), data};
            OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
            OV_ASSERT_NO_THROW(inference_request.infer());
        }
        second_output = inference_request.get_output_tensor(0);

        EXPECT_NE(first_output.data(), second_output.data());
        EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);
    }
}

TEST_P(InferRequestRunTests, CheckOutputDataFromMultipleRunsUsingSameL0Tensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;
    ov::Tensor global_input;

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        global_input = inference_request.get_input_tensor();
        const size_t byte_size = global_input.get_byte_size();
        memset(global_input.data(), 1, byte_size);
        OV_ASSERT_NO_THROW(inference_request.infer());
    }
    first_output = inference_request.get_output_tensor(0);

    for (int i = 0; i < 10; i++) {
        compiled_model = {};
        inference_request = {};

        {
            OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
            OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
            OV_ASSERT_NO_THROW(inference_request.set_input_tensor(global_input));
            OV_ASSERT_NO_THROW(inference_request.infer());
        }
        second_output = inference_request.get_output_tensor(0);

        EXPECT_NE(first_output.data(), second_output.data());
        EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);
    }
}

TEST_P(InferRequestRunTests, RecreateL0TensorIfNeeded) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;
    ov::Tensor global_input;
    float* data;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    global_input = inference_request.get_input_tensor();
    memset(global_input.data(), 1, global_input.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = inference_request.get_output_tensor(0);

    compiled_model = {};
    inference_request = {};

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(global_input));
    OV_ASSERT_NO_THROW(inference_request.infer());
    second_output = inference_request.get_output_tensor(0);

    EXPECT_NE(first_output.data(), second_output.data());
    EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);

    for (int i = 0; i < 10; i++) {
        {
            const size_t byte_size = global_input.get_byte_size();
            data = new float[byte_size / sizeof(float)];
            memset(data, 1, byte_size);
            ov::Tensor input_data_tensor{ov::element::f32, global_input.get_shape(), data};
            OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
            OV_ASSERT_NO_THROW(inference_request.infer());
        }
        second_output = inference_request.get_output_tensor(0);

        EXPECT_NE(first_output.data(), second_output.data());
        EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);

        delete[] data;
    }
}

using RandomTensorOverZeroTensorRunTests = InferRequestRunTests;

TEST_P(RandomTensorOverZeroTensorRunTests, SetRandomTensorOverZeroTensor0) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    auto input_zero_tensor = inference_request.get_input_tensor(0);
    auto* input_zero_data = input_zero_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_zero_data[i] = 5.f;
    }

    inference_request.infer();  // Adds '1' to each element

    auto output_tensor = inference_request.get_output_tensor(0);
    auto* output_data = output_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data[i], 6.f, 1e-5) << "Expected=6, actual=" << output_data[i] << " for index " << i;
    }

    float* buffer = new float[shape_size];
    ov::Tensor tensor{element::f32, shape, buffer};
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 9.f;
    }

    inference_request.set_input_tensor(tensor);
    inference_request.infer();  // Adds '1' to each element
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data[i], 10.f, 1e-5) << "Expected=10, actual=" << output_data[i] << " for index " << i;
    }

    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(input_zero_data[i], 5.f, 1e-5) << "Expected=5, actual=" << input_zero_data[i] << " for index " << i;
    }

    delete[] buffer;
}

TEST_P(RandomTensorOverZeroTensorRunTests, SetRandomTensorOverZeroTensor1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request0, inference_request1;
    inference_request0 = compiled_model.create_infer_request();
    inference_request1 = compiled_model.create_infer_request();

    auto input_zero_tensor = inference_request0.get_input_tensor(0);
    auto* input_zero_data = input_zero_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_zero_data[i] = 5.f;
    }

    inference_request0.infer();  // Adds '1' to each element

    auto output_tensor0 = inference_request0.get_output_tensor(0);
    auto* output_data0 = output_tensor0.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data0[i], 6.f, 1e-5) << "Expected=6, actual=" << output_data0[i] << " for index " << i;
    }

    inference_request1.set_input_tensor(output_tensor0);
    inference_request1.infer();  // Adds '1' to each element

    auto output_tensor1 = inference_request1.get_output_tensor(0);
    auto* output_data1 = output_tensor1.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data1[i], 7.f, 1e-5) << "Expected=7, actual=" << output_data1[i] << " for index " << i;
    }

    float* buffer = new float[shape_size];
    ov::Tensor tensor{element::f32, shape, buffer};
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 9.f;
    }

    inference_request1.set_input_tensor(tensor);
    inference_request1.infer();  // Adds '1' to each element

    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data1[i], 10.f, 1e-5) << "Expected=10, actual=" << output_data1[i] << " for index " << i;
    }

    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(output_data0[i], 6.f, 1e-5) << "Expected=6, actual=" << output_data0[i] << " for index " << i;
    }

    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(input_zero_data[i], 5.f, 1e-5) << "Expected=5, actual=" << input_zero_data[i] << " for index " << i;
    }

    delete[] buffer;
}

using BatchingRunTests = InferRequestRunTests;

TEST_P(BatchingRunTests, CheckBatchingSupportInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    auto batch_shape = Shape{4, 2, 32, 32};
    std::shared_ptr<ov::Model> ov_model_batch = createModel(element::f32, batch_shape, "N...");

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model_batch, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(BatchingRunTests, CheckBatchingSupportAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    auto batch_shape = Shape{4, 2, 32, 32};
    std::shared_ptr<ov::Model> ov_model_batch = createModel(element::f32, batch_shape, "N...");

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model_batch, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(inference_request.start_async());
    inference_request.wait();
}

TEST_P(BatchingRunTests, UseCompilerBatchingErrorPluginBatching) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    std::shared_ptr<ov::Model> ov_model_batch = getDefaultNGraphFunctionForTheDeviceNPU({4, 2, 32, 32});

    auto batch_mode = configuration[ov::intel_npu::batch_mode.name()].as<std::string>();
    try {
        compiled_model = core->compile_model(ov_model_batch, target_device, configuration);
        inference_request = compiled_model.create_infer_request();
        inference_request.start_async();
        inference_request.wait();
    } catch (std::exception& ex) {
        if (batch_mode != "PLUGIN") {
            ASSERT_FALSE(true) << ex.what();
        }
    }
}

TEST_P(BatchingRunTests, SetInputTensorInfer) {
    auto batch_shape = Shape{4, 2, 2, 2};
    auto shape_size = ov::shape_size(batch_shape);
    auto model = createModel(element::f32, batch_shape, "N...");
    float* buffer = new float[shape_size];

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    ov::Tensor tensor{element::f32, batch_shape, buffer};

    inference_request.set_input_tensor(tensor);
    auto actual_tensor = inference_request.get_output_tensor(0);
    auto* actual = actual_tensor.data<float>();
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }

    inference_request.infer();  // Adds '1' to each element
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }
}

TEST_P(BatchingRunTests, SetInputTensorAsync) {
    auto batch_shape = Shape{4, 2, 2, 2};
    auto shape_size = ov::shape_size(batch_shape);
    auto model = createModel(element::f32, batch_shape, "N...");
    float* buffer = new float[shape_size];

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    ov::Tensor tensor{element::f32, batch_shape, buffer};

    inference_request.set_input_tensor(tensor);
    auto actual_tensor = inference_request.get_output_tensor(0);
    auto* actual = actual_tensor.data<float>();
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }

    inference_request.start_async();  // Adds '1' to each element
    inference_request.wait_for(std::chrono::milliseconds(1000));
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }
}

TEST_P(BatchingRunTests, SetInputTensorInfer_Caching) {
    auto batch_shape = Shape{4, 2, 2, 2};
    auto shape_size = ov::shape_size(batch_shape);
    auto model = createModel(element::f32, batch_shape, "N...");
    float* buffer = new float[shape_size];

    m_cache_dir = generateCacheDirName(GetTestName());
    core->set_property({ov::cache_dir(m_cache_dir)});
    auto compiled_model_no_cache = core->compile_model(model, target_device, configuration);
    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    ov::Tensor tensor{element::f32, batch_shape, buffer};

    inference_request.set_input_tensor(tensor);
    auto actual_tensor = inference_request.get_output_tensor(0);
    auto* actual = actual_tensor.data<float>();
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }

    inference_request.infer();  // Adds '1' to each element
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }

    delete[] buffer;
}

TEST_P(BatchingRunTests, CheckTwoRunsInfer) {
    auto batch_shape = Shape{4, 2, 2, 2};
    auto shape_size = ov::shape_size(batch_shape);
    auto model = createModel(element::f32, batch_shape, "N...");
    float* buffer = new float[shape_size];

    auto context = core->get_default_context(target_device);

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    ov::Tensor tensor{element::f32, batch_shape, buffer};

    inference_request.set_input_tensor(tensor);
    auto actual_tensor = inference_request.get_output_tensor(0);
    auto* actual = actual_tensor.data<float>();
    auto* input_data = tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }
    inference_request.infer();  // Adds '1' to each element
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, batch_shape);
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, actual_tensor.get_shape());

    auto* input_data_host_tensor = l0_host_input_tensor.data();
    input_data = reinterpret_cast<float*>(input_data_host_tensor);
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }
    inference_request.set_input_tensor(l0_host_input_tensor);
    inference_request.set_output_tensor(l0_host_output_tensor);
    inference_request.infer();

    auto* actual_host_tensor = l0_host_output_tensor.data();
    actual = reinterpret_cast<float*>(actual_host_tensor);
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }

    delete[] buffer;
}

using RunSeqTests = InferRequestRunTests;

TEST_P(RunSeqTests, CheckMultipleRunsSeq0) {
    auto shape = Shape{1, 64, 64, 256};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    auto context = core->get_default_context(target_device);

    configuration[ov::intel_npu::run_inferences_sequentially.name()] = true;
    configuration[ov::intel_npu::tiles.name()] = 2;
    compiled_model = core->compile_model(model, target_device, configuration);

    const uint32_t inferences = 32;
    std::array<ov::InferRequest, inferences> inference_request;
    ov::Tensor input_tensor;
    std::array<ov::Tensor, inferences> output_tensor;

    input_tensor = context.create_host_tensor(ov::element::f32, shape);
    for (uint32_t i = 0; i < inferences; i++) {
        inference_request[i] = compiled_model.create_infer_request();
        output_tensor[i] = context.create_host_tensor(ov::element::f32, shape);
    }

    inference_request[0].set_input_tensor(input_tensor);
    inference_request[0].set_output_tensor(output_tensor[0]);

    const uint32_t runs = 10;
    for (uint32_t z = 0; z < runs; z++) {
        auto* input_data = reinterpret_cast<float*>(input_tensor.data());
        for (size_t i = 0; i < shape_size; ++i) {
            input_data[i] = static_cast<float>(z);
        }

        inference_request[0].start_async();  // Adds '1' to each element

        for (uint32_t i = 1; i < inferences; i++) {
            inference_request[i].set_input_tensor(output_tensor[i - 1]);
            inference_request[i].set_output_tensor(output_tensor[i]);

            inference_request[i].start_async();  // Adds '1' to each element
        }

        inference_request[inferences - 1].wait();

        float expected_result = static_cast<float>(z) + 1.f;

        for (uint32_t i = 0; i < inferences; i++) {
            auto* output_tensor_data = reinterpret_cast<float*>(output_tensor[i].data());
            for (size_t j = 0; j < shape_size; ++j) {
                EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
                    << "Run=" << z << "Output=" << i << " Expected=" << expected_result
                    << ", actual=" << output_tensor_data[j] << " for index " << j;
            }
            expected_result++;
        }
    }
}

TEST_P(RunSeqTests, CheckMultipleRunsSeq1) {
    auto shape = Shape{1, 64, 64, 256};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    auto context = core->get_default_context(target_device);

    configuration[ov::intel_npu::run_inferences_sequentially.name()] = true;
    configuration[ov::intel_npu::tiles.name()] = 2;
    compiled_model = core->compile_model(model, target_device, configuration);

    const int inferences = 32;
    std::array<ov::InferRequest, inferences> inference_request;
    ov::Tensor input_tensor;
    std::array<ov::Tensor, inferences> output_tensor;

    input_tensor = context.create_host_tensor(ov::element::f32, shape);

    for (int i = 0; i < inferences; i++) {
        inference_request[i] = compiled_model.create_infer_request();
        output_tensor[i] = context.create_host_tensor(ov::element::f32, shape);
    }

    inference_request[inferences - 1].set_input_tensor(input_tensor);
    inference_request[inferences - 1].set_output_tensor(output_tensor[inferences - 1]);

    const int runs = 10;
    for (int z = 0; z < runs; z++) {
        auto* input_data = reinterpret_cast<float*>(input_tensor.data());
        for (size_t i = 0; i < shape_size; ++i) {
            input_data[i] = static_cast<float>(z);
        }

        inference_request[inferences - 1].start_async();  // Adds '1' to each element

        for (int i = inferences - 2; i >= 0; i--) {
            inference_request[i].set_input_tensor(output_tensor[i + 1]);
            inference_request[i].set_output_tensor(output_tensor[i]);

            inference_request[i].start_async();  // Adds '1' to each element
        }

        inference_request[0].wait();

        float expected_result = static_cast<float>(z) + 1.f;

        for (int i = inferences - 1; i >= 0; i--) {
            auto* output_tensor_data = reinterpret_cast<float*>(output_tensor[i].data());
            for (size_t j = 0; j < shape_size; ++j) {
                EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
                    << "Run=" << z << "Output=" << i << " Expected=" << expected_result
                    << ", actual=" << output_tensor_data[j] << " for index " << j;
            }
            expected_result++;
        }
    }
}

TEST_P(RunSeqTests, CheckMultipleRunsSeq2) {
    auto shape = Shape{1, 64, 64, 256};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    auto context = core->get_default_context(target_device);

    configuration[ov::intel_npu::run_inferences_sequentially.name()] = true;
    configuration[ov::intel_npu::tiles.name()] = 2;
    compiled_model = core->compile_model(model, target_device, configuration);

    const int inferences = 32;
    std::array<ov::InferRequest, inferences> inference_request;
    ov::Tensor input_tensor;
    std::array<ov::Tensor, inferences> output_tensor;

    input_tensor = context.create_host_tensor(ov::element::f32, shape);

    for (int i = 0; i < inferences; i++) {
        inference_request[i] = compiled_model.create_infer_request();
        output_tensor[i] = context.create_host_tensor(ov::element::f32, shape);
    }

    inference_request[inferences - 1].set_input_tensor(input_tensor);
    inference_request[inferences - 1].set_output_tensor(output_tensor[inferences - 1]);

    auto* input_data = reinterpret_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 1.f;
    }

    inference_request[inferences - 1].start_async();

    for (int i = inferences - 2; i >= 0; i--) {
        inference_request[i].set_input_tensor(output_tensor[i + 1]);
        inference_request[i].set_output_tensor(output_tensor[i]);

        inference_request[i].start_async();
    }

    inference_request[0].wait();

    try {
        inference_request[5].start_async();
        inference_request[5].wait();
    } catch (const std::exception& ex) {
        ASSERT_FALSE(false) << ex.what();
        return;
    }

    ASSERT_FALSE(true) << "Exception is expected but it didn't throw any exception!";
}

TEST_P(RunSeqTests, CheckMultipleRunsSeq3) {
    auto shape = Shape{1, 64, 64, 256};
    auto model = createModel(element::f32, shape, "N...");

    configuration[ov::intel_npu::run_inferences_sequentially.name()] = true;
    configuration[ov::intel_npu::tiles.name()] = 2;
    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    OV_EXPECT_THROW(inference_request.infer(),
                    ov::Exception,
                    HasSubstr("Only start async is supported when RUN_INFERENCES_SEQUENTIALLY is enabled!"));
}

using BatchingRunSeqTests = InferRequestRunTests;

TEST_P(BatchingRunSeqTests, CheckMultipleBatchingRunsSeq) {
    auto shape = Shape{4, 2, 64, 64};
    auto shape_size = ov::shape_size(shape);
    auto model = createModel(element::f32, shape, "N...");

    auto context = core->get_default_context(target_device);

    configuration[ov::intel_npu::run_inferences_sequentially.name()] = true;
    configuration[ov::intel_npu::tiles.name()] = 2;
    compiled_model = core->compile_model(model, target_device, configuration);

    const uint32_t inferences = 32;
    std::array<ov::InferRequest, inferences> inference_request;
    ov::Tensor input_tensor;
    std::array<ov::Tensor, inferences> output_tensor;

    input_tensor = context.create_host_tensor(ov::element::f32, shape);
    for (uint32_t i = 0; i < inferences; i++) {
        inference_request[i] = compiled_model.create_infer_request();
        output_tensor[i] = context.create_host_tensor(ov::element::f32, shape);
    }

    inference_request[0].set_input_tensor(input_tensor);
    inference_request[0].set_output_tensor(output_tensor[0]);

    const uint32_t runs = 10;
    for (uint32_t z = 0; z < runs; z++) {
        auto* input_data = reinterpret_cast<float*>(input_tensor.data());
        for (size_t i = 0; i < shape_size; ++i) {
            input_data[i] = static_cast<float>(z);
        }

        inference_request[0].start_async();  // Adds '1' to each element

        for (uint32_t i = 1; i < inferences; i++) {
            inference_request[i].set_input_tensor(output_tensor[i - 1]);
            inference_request[i].set_output_tensor(output_tensor[i]);

            inference_request[i].start_async();  // Adds '1' to each element
        }

        inference_request[inferences - 1].wait();

        float expected_result = static_cast<float>(z) + 1.f;

        for (uint32_t i = 0; i < inferences; i++) {
            auto* output_tensor_data = reinterpret_cast<float*>(output_tensor[i].data());
            for (size_t j = 0; j < shape_size; ++j) {
                EXPECT_NEAR(output_tensor_data[j], expected_result, 1e-5)
                    << "Run=" << z << "Output=" << i << " Expected=" << expected_result
                    << ", actual=" << output_tensor_data[j] << " for index " << j;
            }
            expected_result++;
        }
    }
}

using ROITensorInference = OVInferRequestInferenceTests;

TEST_P(ROITensorInference, InferenceROITensor) {
    auto model = OVInferRequestInferenceTests::create_n_inputs(1, ov::element::f32, m_param.m_shape);
    auto compiled_model = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = compiled_model.create_infer_request();
    const std::string tensor_name = "tensor_input0";

    OV_EXPECT_THROW_HAS_SUBSTRING(req.set_tensor(tensor_name, m_param.m_input_tensor),
                                  ov::Exception,
                                  "The tensor is not continuous");
}

using SetShapeInferRunTests = InferRequestRunTests;

TEST_P(SetShapeInferRunTests, checkResultsAfterIOBlobReallocation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto original_shape = Shape{1, 10, 10, 10};
    auto dummy_shape = Shape{1, 50, 100, 100};
    auto shape_size = ov::shape_size(original_shape);
    auto model = createModel(element::f32, original_shape, "N...");

    auto context = core->get_default_context(target_device);

    compiled_model = core->compile_model(model, target_device, configuration);
    ov::InferRequest inference_request;
    inference_request = compiled_model.create_infer_request();

    input = compiled_model.input();
    output = compiled_model.output();

    ov::Tensor input_tensor, first_output_tensor, second_output_tensor;
    auto in_shape = input.get_shape();
    auto out_shape = output.get_shape();

    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 5.f;
    }

    OV_ASSERT_NO_THROW(inference_request.infer());
    OV_ASSERT_NO_THROW(first_output_tensor = inference_request.get_tensor(output));
    // create dummy Tensors to force the driver to allocate memory for the initial tensor somewhere else
    [[maybe_unused]] auto l0_host_dummy_tensor_0 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_1 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_2 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_3 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_4 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_5 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_6 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_7 = context.create_host_tensor(ov::element::f32, dummy_shape);

    auto* actual = first_output_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 6.f, 1e-5) << "Expected=6, actual=" << actual[i] << " for index " << i;
    }

    // imitates blob reallocation
    OV_ASSERT_NO_THROW(input_tensor.set_shape({1, 50, 20, 20}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape(in_shape));

    OV_ASSERT_NO_THROW(second_output_tensor = inference_request.get_tensor(output));
    OV_ASSERT_NO_THROW(second_output_tensor.set_shape({1, 20, 20, 20}));
    OV_ASSERT_NO_THROW(second_output_tensor.set_shape(out_shape));

    OV_ASSERT_NO_THROW(input_tensor = inference_request.get_tensor(input));
    input_data = input_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        input_data[i] = 9.f;
    }

    OV_ASSERT_NO_THROW(inference_request.infer());
    OV_ASSERT_NO_THROW(second_output_tensor = inference_request.get_tensor(output));

    actual = second_output_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_NEAR(actual[i], 10.f, 1e-5) << "Expected=10, actual=" << actual[i] << " for index " << i;
    }
}

TEST_P(SetShapeInferRunTests, checkResultsAfterStateTensorsReallocation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    testing::internal::Random random(1);
    ov::Tensor input_tensor;

    auto original_shape = Shape{1, 10, 10, 10};
    auto dummy_shape = Shape{1, 50, 100, 100};
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

    for (auto&& state : inference_request.query_state()) {
        state.reset();
    }

    OV_ASSERT_NO_THROW(inference_request.infer());

    auto output_tensor = inference_request.get_tensor("sigmod_state");
    auto output_data = output_tensor.data<float>();
    for (size_t i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }

    auto states = inference_request.query_state();
    for (auto state : states) {
        auto last_state = state.get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (size_t i = 0; i < last_state_size; ++i) {
            EXPECT_NEAR(0.0, last_state_data[i], 1e-5);
        }
    }

    // create dummy Tensors to force the driver to allocate memory for the initial tensor somewhere else
    [[maybe_unused]] auto l0_host_dummy_tensor_0 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_1 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_2 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_3 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_4 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_5 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_6 = context.create_host_tensor(ov::element::f32, dummy_shape);
    [[maybe_unused]] auto l0_host_dummy_tensor_7 = context.create_host_tensor(ov::element::f32, dummy_shape);

    for (auto item : inference_request.query_state()) {
        auto tensor_state = item.get_state();
        auto original_shape = tensor_state.get_shape();
        OV_ASSERT_NO_THROW(tensor_state.set_shape({1, 50, 20, 20}));
        OV_ASSERT_NO_THROW(tensor_state.set_shape(original_shape));
    }

    for (auto&& state : inference_request.query_state()) {
        state.reset();
    }

    for (auto state : states) {
        auto last_state = state.get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (size_t i = 0; i < last_state_size; ++i) {
            last_state_data[i] = 1.0f;
        }
    }

    OV_ASSERT_NO_THROW(inference_request.infer());

    for (auto state : states) {
        auto last_state = state.get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (size_t i = 0; i < last_state_size; ++i) {
            EXPECT_NEAR(input_data[i], last_state_data[i], 1e-5);
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
