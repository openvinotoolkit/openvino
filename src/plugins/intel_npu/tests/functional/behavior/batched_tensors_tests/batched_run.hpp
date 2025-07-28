// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/op.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "overload/overload_test_utils_npu.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class BatchedTensorsRunTests : public ov::test::behavior::OVPluginTestBase,
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

    std::shared_ptr<Model> create_n_inputs(size_t n,
                                           element::Type type,
                                           const PartialShape& shape,
                                           const ov::Layout& layout) {
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

// To avoid error: no previous declaration
void executeMutlipleTensorsBatchInfer(ov::InferRequest req,
                                      size_t batch_value,
                                      const Shape& non_batched_shape,
                                      ov::RemoteContext& context);
void executeContiguousTensorBatchInfer(ov::InferRequest req,
                                       size_t batch_value,
                                       const Shape& non_batched_shape,
                                       ov::RemoteContext& context);

// Second test group inheriting from the first
using DynamicBatchedTensorsRunTests = BatchedTensorsRunTests;

TEST_P(BatchedTensorsRunTests, SetInputRemoteTensorsMultipleInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t i = 0; i < batch; ++i) {
        // non contiguous memory
        auto tensor = context.create_host_tensor(ov::element::f32, one_shape);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 5; testNum++) {
        for (size_t i = 0; i < batch; ++i) {
            auto* f = tensors[i].data<float>();
            for (size_t j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20);
            }
        }
        req.infer();  // Adds '1' to each element
        for (size_t j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentTensorsMultipleInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;

    std::vector<float> buffer(one_shape_size * 2 * 2, 0);

    auto tensor0 = ov::Tensor(element::f32, one_shape, &buffer[(0 * 2) * one_shape_size]);
    auto tensor1 = context.create_host_tensor(ov::element::f32, one_shape);
    auto tensor2 = ov::Tensor(element::f32, one_shape, &buffer[(1 * 2) * one_shape_size]);
    auto tensor3 = context.create_host_tensor(ov::element::f32, one_shape);

    tensors.push_back(std::move(tensor0));
    tensors.push_back(std::move(tensor1));
    tensors.push_back(std::move(tensor2));
    tensors.push_back(std::move(tensor3));

    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 5; testNum++) {
        for (size_t i = 0; i < batch; ++i) {
            auto* f = tensors[i].data<float>();
            for (size_t j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20);
            }
        }
        req.infer();  // Adds '1' to each element
        for (size_t j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentTensorsMultipleInferMCL) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();

    std::vector<float> buffer(one_shape_size * batch * 2, 0);

    {
        std::vector<ov::Tensor> tensors;

        auto tensor0 = ov::Tensor(element::f32, one_shape, &buffer[(0 * 2) * one_shape_size]);
        auto tensor1 = context.create_host_tensor(ov::element::f32, one_shape);
        auto tensor2 = ov::Tensor(element::f32, one_shape, &buffer[(1 * 2) * one_shape_size]);
        auto tensor3 = context.create_host_tensor(ov::element::f32, one_shape);

        tensors.push_back(std::move(tensor0));
        tensors.push_back(std::move(tensor1));
        tensors.push_back(std::move(tensor2));
        tensors.push_back(std::move(tensor3));

        req.set_tensors("tensor_input0", tensors);

        auto actual_tensor = req.get_tensor("tensor_output0");
        auto* actual = actual_tensor.data<float>();
        for (auto testNum = 0; testNum < 5; testNum++) {
            for (size_t i = 0; i < batch; ++i) {
                auto* f = tensors[i].data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            req.infer();  // Adds '1' to each element
            for (size_t j = 0; j < one_shape_size * batch; ++j) {
                EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                   << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }

    {
        std::vector<ov::Tensor> tensors;

        auto tensor0 = context.create_host_tensor(ov::element::f32, one_shape);
        auto tensor1 = ov::Tensor(element::f32, one_shape, &buffer[(2 * 2) * one_shape_size]);
        auto tensor2 = ov::Tensor(element::f32, one_shape, &buffer[(3 * 2) * one_shape_size]);
        auto tensor3 = context.create_host_tensor(ov::element::f32, one_shape);

        tensors.push_back(std::move(tensor0));
        tensors.push_back(std::move(tensor1));
        tensors.push_back(std::move(tensor2));
        tensors.push_back(std::move(tensor3));

        req.set_tensors("tensor_input0", tensors);

        auto actual_tensor = req.get_tensor("tensor_output0");
        auto* actual = actual_tensor.data<float>();
        for (auto testNum = 0; testNum < 5; testNum++) {
            for (size_t i = 0; i < batch; ++i) {
                auto* f = tensors[i].data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 200);
                }
            }
            req.infer();  // Adds '1' to each element
            for (size_t j = 0; j < one_shape_size * batch; ++j) {
                EXPECT_EQ(actual[j], testNum + 201) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                    << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentRemoteTensorsMultipleInferMCL) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();

    std::vector<float> buffer(one_shape_size * 2 * 2, 0);

    {
        std::vector<ov::Tensor> tensors;

        auto tensor0 = ov::Tensor(element::f32, one_shape, &buffer[(0 * 2) * one_shape_size]);
        auto tensor1 = context.create_l0_host_tensor(ov::element::f32, one_shape);
        auto tensor2 = context.create_l0_host_tensor(ov::element::f32, one_shape);
        auto tensor3 = context.create_host_tensor(ov::element::f32, one_shape);

        tensors.push_back(tensor0);
        tensors.push_back(tensor1);
        tensors.push_back(tensor2);
        tensors.push_back(tensor3);

        req.set_tensors("tensor_input0", tensors);

        auto actual_tensor = req.get_tensor("tensor_output0");
        auto* actual = actual_tensor.data<float>();
        for (auto testNum = 0; testNum < 5; testNum++) {
            {
                auto* f = tensor0.data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* data = tensor1.get();
                float* f = static_cast<float*>(data);
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* data = tensor2.get();
                float* f = static_cast<float*>(data);
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* f = tensor3.data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }

            req.infer();  // Adds '1' to each element
            for (size_t j = 0; j < one_shape_size * batch; ++j) {
                EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                   << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }

    {
        std::vector<ov::Tensor> tensors;

        auto tensor0 = context.create_l0_host_tensor(ov::element::f32, one_shape);
        auto tensor1 = context.create_host_tensor(ov::element::f32, one_shape);
        auto tensor2 = ov::Tensor(element::f32, one_shape, &buffer[(1 * 2) * one_shape_size]);
        auto tensor3 = context.create_l0_host_tensor(ov::element::f32, one_shape);

        tensors.push_back(tensor0);
        tensors.push_back(tensor1);
        tensors.push_back(tensor2);
        tensors.push_back(tensor3);

        req.set_tensors("tensor_input0", tensors);

        auto actual_tensor = req.get_tensor("tensor_output0");
        auto* actual = actual_tensor.data<float>();
        for (auto testNum = 0; testNum < 5; testNum++) {
            {
                auto* data = tensor0.get();
                float* f = static_cast<float*>(data);
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* f = tensor1.data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* f = tensor2.data<float>();
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }
            {
                auto* data = tensor3.get();
                float* f = static_cast<float*>(data);
                for (size_t j = 0; j < one_shape_size; ++j) {
                    f[j] = static_cast<float>(testNum + 20);
                }
            }

            req.infer();  // Adds '1' to each element
            for (size_t j = 0; j < one_shape_size * batch; ++j) {
                EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                   << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsMultipleInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto modelShape = PartialShape{ov::Dimension(1, 10), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t i = 0; i < batch; ++i) {
        // non contiguous memory
        auto tensor = context.create_host_tensor(ov::element::f32, one_shape);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 1; testNum++) {
        for (size_t i = 0; i < batch; ++i) {
            auto* f = tensors[i].data<float>();
            for (size_t j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20 * (i + 1));
            }
        }
        req.infer();  // Adds '1' to each element
        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < one_shape_size; ++j) {
                auto expected = testNum + 20 * (i + 1) + 1;
                EXPECT_EQ(actual[i * one_shape_size + j], expected)
                    << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index "
                    << j << ", batch: " << i;
            }
        }
    }
}

void executeMutlipleTensorsBatchInfer(ov::InferRequest req,
                                      size_t batch_value,
                                      const Shape& non_batched_shape,
                                      ov::RemoteContext& context) {
    auto non_batched_shape_size = ov::shape_size(non_batched_shape);
    std::vector<ov::Tensor> tensors;
    tensors.reserve(batch_value);
    ASSERT_TRUE(batch_value != 0);
    for (size_t i = 0; i < batch_value; ++i) {
        // non contiguous memory
        auto tensor = context.create_host_tensor(ov::element::f32, non_batched_shape);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    ASSERT_EQ(actual_tensor.get_byte_size(), tensors.back().get_byte_size() * tensors.size())
        << "\"tensor_output0\" must have the same size as \"tensor_input0\" for batch_value value:" << batch_value;
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 1; testNum++) {
        for (size_t i = 0; i < batch_value; ++i) {
            auto* f = tensors[i].data<float>();
            for (size_t j = 0; j < non_batched_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20 * (i + 1));
            }
        }
        req.infer();
        for (size_t i = 0; i < batch_value; i++) {
            for (size_t j = 0; j < non_batched_shape_size; ++j) {
                auto expected = testNum + 20 * (i + 1) + 1;
                EXPECT_EQ(actual[i * non_batched_shape_size + j], expected)
                    << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index "
                    << j << ", batch: " << i;
            }
        }
    }
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsDynamicBatchInflation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(model_batch_bottom_bound, model_batch_upper_bound), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t tensor_batch = model_batch_bottom_bound; tensor_batch <= model_batch_upper_bound; ++tensor_batch) {
        executeMutlipleTensorsBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsDynamicBatchDeflation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(model_batch_bottom_bound, model_batch_upper_bound), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t tensor_batch = model_batch_upper_bound; tensor_batch >= model_batch_bottom_bound; tensor_batch--) {
        executeMutlipleTensorsBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorSingleInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    size_t tensor_batch = 3;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto tensor_shape = Shape{tensor_batch, 2, 2, 2};
    auto tensor_shape_size = ov::shape_size(tensor_shape);
    auto modelShape = PartialShape{ov::Dimension(1, 10), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest

    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    // contiguous memory
    auto tensor = context.create_host_tensor(ov::element::f32, tensor_shape);
    tensors.push_back(std::move(tensor));

    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 5; testNum++) {
        auto* f = tensors[0].data<float>();
        for (size_t j = 0; j < tensor_shape_size; ++j) {
            f[j] = static_cast<float>(testNum + 20 * (j / one_shape_size + 1));
        }
        req.infer();  // Adds '1' to each element
        for (size_t j = 0; j < tensor_shape_size; ++j) {
            auto expected = testNum + 20 * (j / one_shape_size + 1) + 1;
            EXPECT_EQ(actual[j], expected)
                << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index " << j
                << ", batch: " << j / one_shape_size;
        }
    }
}

void executeContiguousTensorBatchInfer(ov::InferRequest req,
                                       size_t batch_value,
                                       const Shape& non_batched_shape,
                                       ov::RemoteContext& context) {
    auto non_batched_shape_size = ov::shape_size(non_batched_shape);
    auto tensor_shape = non_batched_shape;
    tensor_shape[0] = batch_value;
    auto tensor_shape_size = ov::shape_size(tensor_shape);
    auto tensor = context.create_host_tensor(ov::element::f32, tensor_shape);

    std::vector<ov::Tensor> tensors;
    tensors.push_back(std::move(tensor));

    req.set_tensors("tensor_input0", tensors);
    auto actual_tensor = req.get_tensor("tensor_output0");
    ASSERT_EQ(actual_tensor.get_byte_size(), tensors.back().get_byte_size())
        << "\"tensor_output0\" must have the same size as \"tensor_input0\" for batch_value value:" << batch_value;

    auto* actual = actual_tensor.data<float>();
    auto* f = tensors.back().data<float>();
    for (size_t j = 0; j < tensor_shape_size; ++j) {
        f[j] = static_cast<float>(batch_value + 20 * (j / non_batched_shape_size + 1));
    }
    req.infer();
    // check that we got valid inference results on each lines of N
    for (size_t j = 0; j < tensor_shape_size; ++j) {
        auto expected = batch_value + 20 * (j / non_batched_shape_size + 1) + 1;
        EXPECT_EQ(actual[j], expected) << "Infer " << batch_value << ": Expected=" << expected
                                       << ", actual=" << actual[j] << " for index " << j
                                       << ", batch: " << j / non_batched_shape_size;
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorDynamicBatchInflation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(model_batch_bottom_bound, model_batch_upper_bound), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);

    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t tensor_batch = model_batch_bottom_bound; tensor_batch <= model_batch_upper_bound; tensor_batch++) {
        // dynamically change N of contiduous memory tensor in [model_batch_bottom_bound..model_batch_upper_bound]
        // and check that the output tensor kept modified accordingly
        executeContiguousTensorBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorDynamicBatchDeflation) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(model_batch_bottom_bound, model_batch_upper_bound), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);

    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t tensor_batch = model_batch_upper_bound; tensor_batch >= model_batch_bottom_bound; tensor_batch--) {
        // dynamically change N of contiduous memory tensor in [model_batch_bottom_bound..model_batch_upper_bound]
        // and check that the output tensor kept modified accordingly
        executeContiguousTensorBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputDifferentTensorsMultipleInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto modelShape = PartialShape{ov::Dimension(1, 10), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;

    std::vector<float> buffer(one_shape_size * 2 * 2, 0);

    auto tensor0 = ov::Tensor(element::f32, one_shape, &buffer[(0 * 2) * one_shape_size]);
    auto tensor1 = context.create_host_tensor(ov::element::f32, one_shape);
    auto tensor2 = ov::Tensor(element::f32, one_shape, &buffer[(1 * 2) * one_shape_size]);
    auto tensor3 = context.create_host_tensor(ov::element::f32, one_shape);

    tensors.push_back(std::move(tensor0));
    tensors.push_back(std::move(tensor1));
    tensors.push_back(std::move(tensor2));
    tensors.push_back(std::move(tensor3));

    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 1; testNum++) {
        for (size_t i = 0; i < batch; ++i) {
            auto* f = tensors[i].data<float>();
            for (size_t j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20);
            }
        }
        req.infer();  // Adds '1' to each element
        for (size_t j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
