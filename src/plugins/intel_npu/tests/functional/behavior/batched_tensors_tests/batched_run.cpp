// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
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
class BatchedTensorsRunTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
    std::string m_cache_dir;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

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
            ov::test::utils::removeFilesWithExt<opt::FORCE>(m_cache_dir, "blob");
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

    static constexpr size_t DYNAMIC_BATCH_LOWER_BOUND = 1;
    static constexpr size_t DYNAMIC_BATCH_UPPER_BOUND = 10;
    // The deflation loops of the tests count down with an unsigned index: a lower bound of 0 would never end them
    static_assert(DYNAMIC_BATCH_LOWER_BOUND >= 1, "The batch lower bound must be at least 1");

    // Creates a request for a model with two independent inputs (tensor_input0/1, each one adding '1' to its own
    // output tensor_output0/1) with a dynamic batch in [DYNAMIC_BATCH_LOWER_BOUND, DYNAMIC_BATCH_UPPER_BOUND]
    ov::InferRequest createTwoInputsDynamicBatchRequest() {
        auto model = create_n_inputs(2, element::f32, Shape{4, 2, 2, 2}, "N...");
        const auto modelShape =
            PartialShape{ov::Dimension(DYNAMIC_BATCH_LOWER_BOUND, DYNAMIC_BATCH_UPPER_BOUND), 2, 2, 2};
        std::map<std::string, ov::PartialShape> shapes;
        shapes["tensor_input0"] = modelShape;
        shapes["tensor_input1"] = modelShape;
        model->reshape(shapes);
        return core->compile_model(model, target_device, configuration).create_infer_request();
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

TEST_P(BatchedTensorsRunTests, SetInputRemoteTensorsMultipleInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
            ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentTensorsMultipleInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
            ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputTensorsWithMismatchedNonBatchDimension) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 2;
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    auto make_tensors = [&](const Shape& one_shape) {
        std::vector<ov::Tensor> tensors;
        for (size_t i = 0; i < batch; ++i) {
            tensors.push_back(ov::Tensor(element::f32, one_shape));
        }
        return tensors;
    };

    // The tensors agree with the model on the batch dimension and with each other, but not on the remaining
    // dimensions. They must be rejected by set_tensors: the Level Zero buffer backing this input is allocated
    // from the compiler shape, so copying these tensors into it writes far past the end of that allocation.
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors({1, 2, 2, 512})), ov::Exception);
    // Smaller non-batch dimension: the device would read past the end of the user buffers
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors({1, 2, 2, 1})), ov::Exception);
    // Rank mismatch
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors({1, 2, 2, 2, 2})), ov::Exception);
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors({1, 2, 2})), ov::Exception);
    // Scalar tensors have no batch dimension to index
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors({})), ov::Exception);
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentTensorsMultipleInferWithExportImport) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    // Compile model
    auto execNet = core->compile_model(model, target_device, configuration);

    // Export model
    std::stringstream modelStream;
    execNet.export_model(modelStream);

    // Import model
    auto importedExecNet = core->import_model(modelStream, target_device, configuration);

    auto context = core->get_default_context(target_device);

    // Create InferRequest from imported model
    ov::InferRequest req;
    req = importedExecNet.create_infer_request();
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
            ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentTensorsMultipleInferMCL) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{batch, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    auto context = core->get_default_context(target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();

    size_t buffer_size = one_shape_size * batch * 2;
    float* buffer = static_cast<float*>(::operator new(buffer_size * sizeof(float), std::align_val_t(4096)));
    std::memset(buffer, 0, buffer_size * sizeof(float));

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
                ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
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
                ASSERT_EQ(actual[j], testNum + 201) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                    << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }

    ::operator delete(buffer, std::align_val_t(4096));
}

TEST_P(BatchedTensorsRunTests, checkResultsAfterRunningWithSameRawMemoryMultipleTimes) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{batch, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    float* buffer0 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer1 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer2 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer3 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));

    std::vector<ov::Tensor> tensors;
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer0));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer1));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer2));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer3));

    req.set_tensors("tensor_input0", tensors);

    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < one_shape_size; ++j) {
            buffer0[j] = static_cast<float>(i);
            buffer1[j] = static_cast<float>(i);
            buffer2[j] = static_cast<float>(i);
            buffer3[j] = static_cast<float>(i);
        }

        req.infer();

        auto output_tensor = req.get_tensor("tensor_output0");
        auto* output_data = output_tensor.data<float>();
        for (size_t j = 0; j < one_shape_size * batch; ++j) {
            ASSERT_EQ(output_data[j], static_cast<float>(i) + 1.0f)
                << "Run " << i << ": Expected=" << static_cast<float>(i) + 1.0f << ", actual=" << output_data[j]
                << " for index " << j;
        }
    }

    ::operator delete(buffer0, std::align_val_t(4096));
    ::operator delete(buffer1, std::align_val_t(4096));
    ::operator delete(buffer2, std::align_val_t(4096));
    ::operator delete(buffer3, std::align_val_t(4096));
}

TEST_P(BatchedTensorsRunTests, checkResultsAfterRunningWithRawAlignedMemoryWithSetTensorAndTensors) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{4, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto batch_size = ov::shape_size(batch_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    float* bigger_buffer = static_cast<float*>(::operator new(batch_size * sizeof(float), std::align_val_t(4096)));
    for (size_t j = 0; j < batch_size; ++j) {
        bigger_buffer[j] = 1.0f;
    }

    ov::Tensor bigger_tensor(element::f32, batch_shape, bigger_buffer);
    req.set_tensor("tensor_input0", bigger_tensor);
    req.infer();

    ::operator delete(bigger_buffer, std::align_val_t(4096));

    auto output_tensor = req.get_tensor("tensor_output0");
    auto* output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 2.0f) << "Expected=" << 2.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    float* buffer0 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer1 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer2 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer3 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));

    std::vector<ov::Tensor> tensors;
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer0));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer1));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer2));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer3));
    req.set_tensors("tensor_input0", tensors);

    for (size_t j = 0; j < one_shape_size; ++j) {
        buffer0[j] = 5.0f;
        buffer1[j] = 5.0f;
        buffer2[j] = 5.0f;
        buffer3[j] = 5.0f;
    }

    req.infer();

    output_tensor = req.get_tensor("tensor_output0");
    output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 6.0f) << "Expected=" << 6.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    ::operator delete(buffer0, std::align_val_t(4096));
    ::operator delete(buffer1, std::align_val_t(4096));
    ::operator delete(buffer2, std::align_val_t(4096));
    ::operator delete(buffer3, std::align_val_t(4096));
}

TEST_P(BatchedTensorsRunTests, checkResultsAfterRunningWithRawAlignedMemoryWithSetTensorsAndTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{4, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto batch_size = ov::shape_size(batch_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    float* buffer0 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer1 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer2 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer3 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));

    std::vector<ov::Tensor> tensors;
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer0));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer1));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer2));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer3));
    req.set_tensors("tensor_input0", tensors);

    for (size_t j = 0; j < one_shape_size; ++j) {
        buffer0[j] = 5.0f;
        buffer1[j] = 5.0f;
        buffer2[j] = 5.0f;
        buffer3[j] = 5.0f;
    }

    req.infer();

    auto output_tensor = req.get_tensor("tensor_output0");
    auto* output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 6.0f) << "Expected=" << 6.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    ::operator delete(buffer0, std::align_val_t(4096));
    ::operator delete(buffer1, std::align_val_t(4096));
    ::operator delete(buffer2, std::align_val_t(4096));
    ::operator delete(buffer3, std::align_val_t(4096));

    float* bigger_buffer = static_cast<float*>(::operator new(batch_size * sizeof(float), std::align_val_t(4096)));
    for (size_t j = 0; j < batch_size; ++j) {
        bigger_buffer[j] = 1.0f;
    }

    ov::Tensor bigger_tensor(element::f32, batch_shape, bigger_buffer);
    req.set_tensor("tensor_input0", bigger_tensor);
    req.infer();

    ::operator delete(bigger_buffer, std::align_val_t(4096));

    output_tensor = req.get_tensor("tensor_output0");
    output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 2.0f) << "Expected=" << 2.0f << ", actual=" << output_data[j] << " for index " << j;
    }
}

TEST_P(BatchedTensorsRunTests, checkResultsAfterRunningWithRawUnalignedMemoryWithSetTensorAndTensors) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{4, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto batch_size = ov::shape_size(batch_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    float* bigger_buffer =
        static_cast<float*>(::operator new(batch_size * sizeof(float) + 16 * sizeof(float), std::align_val_t(4096)));
    for (size_t j = 0; j < batch_size + 16; ++j) {
        bigger_buffer[j] = 1.0f;
    }

    ov::Tensor bigger_tensor(element::f32, batch_shape, bigger_buffer + 16);
    req.set_tensor("tensor_input0", bigger_tensor);
    req.infer();

    ::operator delete(bigger_buffer, std::align_val_t(4096));

    auto output_tensor = req.get_tensor("tensor_output0");
    auto* output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 2.0f) << "Expected=" << 2.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    float* buffer0 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer1 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer2 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer3 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));

    std::vector<ov::Tensor> tensors;
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer0));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer1));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer2));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer3));
    req.set_tensors("tensor_input0", tensors);

    for (size_t j = 0; j < one_shape_size; ++j) {
        buffer0[j] = 5.0f;
        buffer1[j] = 5.0f;
        buffer2[j] = 5.0f;
        buffer3[j] = 5.0f;
    }

    req.infer();

    output_tensor = req.get_tensor("tensor_output0");
    output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 6.0f) << "Expected=" << 6.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    ::operator delete(buffer0, std::align_val_t(4096));
    ::operator delete(buffer1, std::align_val_t(4096));
    ::operator delete(buffer2, std::align_val_t(4096));
    ::operator delete(buffer3, std::align_val_t(4096));
}

TEST_P(BatchedTensorsRunTests, checkResultsAfterRunningWithRawUnalignedMemoryWithSetTensorsAndTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto one_shape = Shape{1, 16, 16, 16};
    auto batch_shape = Shape{4, 16, 16, 16};
    auto one_shape_size = ov::shape_size(one_shape);
    auto batch_size = ov::shape_size(batch_shape);
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    float* buffer0 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer1 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer2 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));
    float* buffer3 = static_cast<float*>(::operator new(one_shape_size * sizeof(float), std::align_val_t(4096)));

    std::vector<ov::Tensor> tensors;
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer0));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer1));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer2));
    tensors.push_back(ov::Tensor(element::f32, one_shape, buffer3));
    req.set_tensors("tensor_input0", tensors);

    for (size_t j = 0; j < one_shape_size; ++j) {
        buffer0[j] = 5.0f;
        buffer1[j] = 5.0f;
        buffer2[j] = 5.0f;
        buffer3[j] = 5.0f;
    }

    req.infer();

    auto output_tensor = req.get_tensor("tensor_output0");
    auto* output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 6.0f) << "Expected=" << 6.0f << ", actual=" << output_data[j] << " for index " << j;
    }

    ::operator delete(buffer0, std::align_val_t(4096));
    ::operator delete(buffer1, std::align_val_t(4096));
    ::operator delete(buffer2, std::align_val_t(4096));
    ::operator delete(buffer3, std::align_val_t(4096));

    float* bigger_buffer =
        static_cast<float*>(::operator new(batch_size * sizeof(float) + 16 * sizeof(float), std::align_val_t(4096)));
    for (size_t j = 0; j < batch_size + 16; ++j) {
        bigger_buffer[j] = 1.0f;
    }

    ov::Tensor bigger_tensor(element::f32, batch_shape, bigger_buffer + 16);
    req.set_tensor("tensor_input0", bigger_tensor);
    req.infer();

    ::operator delete(bigger_buffer, std::align_val_t(4096));

    output_tensor = req.get_tensor("tensor_output0");
    output_data = output_tensor.data<float>();
    for (size_t j = 0; j < batch_size; ++j) {
        ASSERT_EQ(output_data[j], 2.0f) << "Expected=" << 2.0f << ", actual=" << output_data[j] << " for index " << j;
    }
}

TEST_P(BatchedTensorsRunTests, SetInputDifferentRemoteTensorsMultipleInferMCL) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
                ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
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
                ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                                   << ", actual=" << actual[j] << " for index " << j;
            }
        }
    }
}

using DynamicBatchedTensorsRunTests = BatchedTensorsRunTests;

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsMultipleInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
                ASSERT_EQ(actual[i * one_shape_size + j], expected)
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
    std::vector<ov::Tensor> tensors0, tensors1;
    tensors0.reserve(batch_value);
    tensors1.reserve(batch_value);
    ASSERT_TRUE(batch_value != 0);
    for (size_t i = 0; i < batch_value; ++i) {
        // non contiguous memory
        auto tensor0 = context.create_host_tensor(ov::element::f32, non_batched_shape);
        auto tensor1 = context.create_host_tensor(ov::element::f32, non_batched_shape);
        tensors0.push_back(std::move(tensor0));
        tensors1.push_back(std::move(tensor1));
    }

    req.set_tensors("tensor_input0", tensors0);
    req.set_tensors("tensor_input1", tensors1);

    auto actual_tensor = req.get_tensor("tensor_output0");
    ASSERT_EQ(actual_tensor.get_byte_size(), tensors0.back().get_byte_size() * tensors0.size())
        << "\"tensor_output0\" must have the same size as \"tensor_input0\" for batch_value value:" << batch_value;
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 1; testNum++) {
        for (size_t i = 0; i < batch_value; ++i) {
            auto* f = tensors0[i].data<float>();
            for (size_t j = 0; j < non_batched_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20 * (i + 1));
            }
        }
        req.infer();
        for (size_t i = 0; i < batch_value; i++) {
            for (size_t j = 0; j < non_batched_shape_size; ++j) {
                auto expected = testNum + 20 * (i + 1) + 1;
                ASSERT_EQ(actual[i * non_batched_shape_size + j], expected)
                    << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index "
                    << j << ", batch: " << i;
            }
        }
    }
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsDynamicBatchInflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsDynamicUnboundedBatchInflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(-1), 2, 2, 2};
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

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputRemoteTensorsDynamicUnboundedBatchDeflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    size_t model_batch_bottom_bound = 1;
    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(-1), 2, 2, 2};
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
            ASSERT_EQ(actual[j], expected)
                << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index " << j
                << ", batch: " << j / one_shape_size;
        }
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorSingleInferWithExportImport) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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

    // Compile model
    auto execNet = core->compile_model(model, target_device, configuration);

    // Export model
    std::stringstream modelStream;
    execNet.export_model(modelStream);

    // Import model
    auto importedExecNet = core->import_model(modelStream, target_device, configuration);

    auto context = core->get_default_context(target_device);

    // Create InferRequest from imported model
    ov::InferRequest req;
    req = importedExecNet.create_infer_request();
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
            ASSERT_EQ(actual[j], expected)
                << "Infer " << testNum << ": Expected=" << expected << ", actual=" << actual[j] << " for index " << j
                << ", batch: " << j / one_shape_size;
        }
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleUnboundedBatchedTensorSingleInferWithExportImport) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t batch = 4;
    size_t tensor_batch = 3;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto tensor_shape = Shape{tensor_batch, 2, 2, 2};
    auto tensor_shape_size = ov::shape_size(tensor_shape);
    auto modelShape = PartialShape{ov::Dimension(-1), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(2, element::f32, batch_shape, "N...");

    const std::string tensor_name_0 = "tensor_input0";
    const std::string tensor_name_1 = "tensor_input1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name_0] = modelShape;
    shapes[tensor_name_1] = modelShape;
    model->reshape(shapes);

    // Compile model
    auto execNet = core->compile_model(model, target_device, configuration);

    // Export model
    std::stringstream modelStream;
    execNet.export_model(modelStream);

    // Import model
    auto importedExecNet = core->import_model(modelStream, target_device, configuration);

    auto context = core->get_default_context(target_device);

    // Create InferRequest from imported model
    ov::InferRequest req;
    req = importedExecNet.create_infer_request();
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
            ASSERT_EQ(actual[j], expected)
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
    auto tensorIn0 = context.create_host_tensor(ov::element::f32, tensor_shape);
    auto tensorIn1 = context.create_host_tensor(ov::element::f32, tensor_shape);

    std::vector<ov::Tensor> tensors0, tensors1;
    tensors0.push_back(std::move(tensorIn0));
    tensors1.push_back(std::move(tensorIn1));

    req.set_tensors("tensor_input0", tensors0);
    req.set_tensors("tensor_input1", tensors1);
    auto actual_tensor = req.get_tensor("tensor_output0");
    ASSERT_EQ(actual_tensor.get_byte_size(), tensors0.back().get_byte_size())
        << "\"tensor_output0\" must have the same size as \"tensor_input0\" for batch_value value:" << batch_value;

    auto* actual = actual_tensor.data<float>();
    auto* f = tensors0.back().data<float>();
    for (size_t j = 0; j < tensor_shape_size; ++j) {
        f[j] = static_cast<float>(batch_value + 20 * (j / non_batched_shape_size + 1));
    }
    req.infer();
    // check that we got valid inference results on each lines of N
    for (size_t j = 0; j < tensor_shape_size; ++j) {
        auto expected = batch_value + 20 * (j / non_batched_shape_size + 1) + 1;
        ASSERT_EQ(actual[j], expected) << "Infer " << batch_value << ": Expected=" << expected
                                       << ", actual=" << actual[j] << " for index " << j
                                       << ", batch: " << j / non_batched_shape_size;
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorDynamicBatchInflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
    for (size_t tensor_batch = model_batch_bottom_bound; tensor_batch <= model_batch_upper_bound; tensor_batch++) {
        // dynamically change N of contiguous memory tensor in [model_batch_bottom_bound..model_batch_upper_bound]
        // and check that the output tensor kept modified accordingly
        executeContiguousTensorBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorDynamicBatchInflationWithExportImport) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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

    // Compile model
    auto execNet = core->compile_model(model, target_device, configuration);

    // Export model
    std::stringstream modelStream;
    execNet.export_model(modelStream);

    // Import model
    auto importedExecNet = core->import_model(modelStream, target_device, configuration);

    auto context = core->get_default_context(target_device);

    // Create InferRequest from imported model
    ov::InferRequest req;
    req = importedExecNet.create_infer_request();

    for (size_t tensor_batch = model_batch_bottom_bound; tensor_batch <= model_batch_upper_bound; tensor_batch++) {
        // dynamically change N of contiguous memory tensor in [model_batch_bottom_bound..model_batch_upper_bound]
        // and check that the output tensor kept modified accordingly
        executeContiguousTensorBatchInfer(req, tensor_batch, one_shape, context);
    }
}

TEST_P(DynamicBatchedTensorsRunTests, SetInputRemoteSingleBatchedTensorDynamicBatchDeflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
    for (size_t tensor_batch = model_batch_upper_bound; tensor_batch >= model_batch_bottom_bound; tensor_batch--) {
        // dynamically change N of contiguous memory tensor in [model_batch_bottom_bound..model_batch_upper_bound]
        // and check that the output tensor kept modified accordingly
        executeContiguousTensorBatchInfer(req, tensor_batch, one_shape, context);
    }
}

namespace {

// Fills both inputs with different values, infers and checks that both outputs have the new batch size and values
void fillInferAndCheckTwoInputs(ov::InferRequest& req, ov::Tensor& input0, ov::Tensor& input1, const float value) {
    auto* data0 = input0.data<float>();
    for (size_t i = 0; i < input0.get_size(); ++i) {
        data0[i] = value;
    }
    auto* data1 = input1.data<float>();
    for (size_t i = 0; i < input1.get_size(); ++i) {
        data1[i] = value + 50.f;
    }

    req.infer();  // Adds '1' to each element

    // Outputs can be reallocated when the batch changes, so always query them after the inference
    auto output0 = req.get_tensor("tensor_output0");
    auto output1 = req.get_tensor("tensor_output1");
    ASSERT_EQ(output0.get_shape(), input0.get_shape());
    ASSERT_EQ(output1.get_shape(), input1.get_shape());

    const auto* out0 = output0.data<float>();
    for (size_t j = 0; j < output0.get_size(); ++j) {
        ASSERT_EQ(out0[j], value + 1.f) << "Batch=" << input0.get_shape()[0] << ", output0 index " << j;
    }
    const auto* out1 = output1.data<float>();
    for (size_t j = 0; j < output1.get_size(); ++j) {
        ASSERT_EQ(out1[j], value + 51.f) << "Batch=" << input1.get_shape()[0] << ", output1 index " << j;
    }
}

// Sets both inputs and runs once, so the request has a pipeline for their batch size
void setInputsAndInferOnce(ov::InferRequest& req, ov::Tensor& input0, ov::Tensor& input1) {
    req.set_tensor("tensor_input0", input0);
    req.set_tensor("tensor_input1", input1);
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 1.f));
}

}  // namespace

// Both inputs are reshaped (set_shape) without calling set_tensor again. The plugin must detect the new batch size
// and sync the Level Zero tensors of both inputs, not only of the first one.
TEST_P(DynamicBatchedTensorsRunTests, ReshapeAllInputTensorsWithoutSetTensorInflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    for (size_t batch = 3; batch <= DYNAMIC_BATCH_UPPER_BOUND; ++batch) {
        input0.set_shape(Shape{batch, 2, 2, 2});
        input1.set_shape(Shape{batch, 2, 2, 2});
        ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, static_cast<float>(batch)));
    }
}

TEST_P(DynamicBatchedTensorsRunTests, ReshapeAllInputTensorsWithoutSetTensorDeflation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{DYNAMIC_BATCH_UPPER_BOUND, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{DYNAMIC_BATCH_UPPER_BOUND, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    for (size_t batch = DYNAMIC_BATCH_UPPER_BOUND - 1; batch >= DYNAMIC_BATCH_LOWER_BOUND && batch > 0; --batch) {
        input0.set_shape(Shape{batch, 2, 2, 2});
        input1.set_shape(Shape{batch, 2, 2, 2});
        ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, static_cast<float>(batch)));
    }
}

// Both inputs are reshaped and set_tensor is called again for both of them (same tensor objects)
TEST_P(DynamicBatchedTensorsRunTests, ReshapeAllInputTensorsThenSetTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    for (size_t batch = 3; batch <= DYNAMIC_BATCH_UPPER_BOUND; ++batch) {
        input0.set_shape(Shape{batch, 2, 2, 2});
        input1.set_shape(Shape{batch, 2, 2, 2});
        req.set_tensor("tensor_input0", input0);
        req.set_tensor("tensor_input1", input1);
        ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, static_cast<float>(batch)));
    }
}

// Running twice after a reshape must not trigger any update the second time and must keep producing valid results
TEST_P(DynamicBatchedTensorsRunTests, ReshapeAllInputTensorsAndInferTwice) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    input0.set_shape(Shape{6, 2, 2, 2});
    input1.set_shape(Shape{6, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 10.f));
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 20.f));
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 30.f));
}

// After a batch-2 inference, one input is replaced with a batch-5 tensor (set_tensor) and the other one is reshaped to
// batch 5 (set_shape) without calling its setter. The set_tensor call already updates the graph batch size, but the
// Level Zero tensor of the reshaped input must still be synced.
TEST_P(DynamicBatchedTensorsRunTests, SetTensorOnFirstInputAndReshapeSecondInput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    ov::Tensor newInput0(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input0", newInput0);
    input1.set_shape(Shape{5, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, newInput0, input1, 2.f));

    // Same thing again with bigger and smaller batches, alternating which input is replaced
    ov::Tensor newInput1(ov::element::f32, Shape{8, 2, 2, 2});
    req.set_tensor("tensor_input1", newInput1);
    newInput0.set_shape(Shape{8, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, newInput0, newInput1, 3.f));

    ov::Tensor smallerInput0(ov::element::f32, Shape{3, 2, 2, 2});
    req.set_tensor("tensor_input0", smallerInput0);
    newInput1.set_shape(Shape{3, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, smallerInput0, newInput1, 4.f));
}

// The reverse order: the first input is reshaped, the second one is replaced with set_tensor
TEST_P(DynamicBatchedTensorsRunTests, ReshapeFirstInputAndSetTensorOnSecondInput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    input0.set_shape(Shape{5, 2, 2, 2});
    ov::Tensor newInput1(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input1", newInput1);
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, newInput1, 2.f));
}

// Only one of the two inputs is reshaped, so the batch sizes of the inputs no longer match: an error is expected
TEST_P(DynamicBatchedTensorsRunTests, ReshapeOnlyOneInputTensorFails) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    input0.set_shape(Shape{5, 2, 2, 2});
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    // The error is reported again and the request state is not corrupted by the failed inference
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    // Updating the remaining input fixes the request
    input1.set_shape(Shape{5, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 2.f));
}

// Only one of the two inputs gets a new tensor (set_tensor) with another batch size: same error as for set_shape
TEST_P(DynamicBatchedTensorsRunTests, SetTensorOnlyOneInputTensorFails) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    ov::Tensor newInput0(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input0", newInput0);
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    // Setting the other input with the same batch size fixes the request
    ov::Tensor newInput1(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input1", newInput1);
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, newInput0, newInput1, 3.f));
}

namespace {

// Checks that all the samples of the batch were computed: sample i of "tensor_output0" must be (i + 2) and
// "tensor_output1" must be 101, given that the inputs were filled with (i + 1) and 100
void checkAllSamplesWereComputed(ov::InferRequest& req, const size_t batch, const Shape& oneSampleShape) {
    auto expectedShape = oneSampleShape;
    expectedShape[0] = batch;
    const auto oneSampleSize = ov::shape_size(oneSampleShape);

    auto output0 = req.get_tensor("tensor_output0");
    auto output1 = req.get_tensor("tensor_output1");
    ASSERT_EQ(output0.get_shape(), expectedShape);
    ASSERT_EQ(output1.get_shape(), expectedShape);

    const auto* out0 = output0.data<float>();
    const auto* out1 = output1.data<float>();
    for (size_t j = 0; j < oneSampleSize * batch; ++j) {
        const auto sample = j / oneSampleSize;
        ASSERT_EQ(out0[j], static_cast<float>(sample + 2)) << "output0, sample " << sample << ", index " << j;
        ASSERT_EQ(out1[j], 101.f) << "output1, sample " << sample << ", index " << j;
    }
}

// Inputs for the setter order tests: input0 as a list of tensors (set_tensors), input1 as one batched tensor
struct SetterOrderInputs {
    std::vector<ov::Tensor> input0Tensors;
    ov::Tensor input1;

    SetterOrderInputs(const size_t batch, const Shape& oneSampleShape) {
        auto batchShape = oneSampleShape;
        batchShape[0] = batch;
        for (size_t i = 0; i < batch; ++i) {
            input0Tensors.emplace_back(ov::element::f32, oneSampleShape);
        }
        input1 = ov::Tensor(ov::element::f32, batchShape);
    }

    void fill() {
        for (size_t i = 0; i < input0Tensors.size(); ++i) {
            auto* data = input0Tensors[i].data<float>();
            std::fill(data, data + input0Tensors[i].get_size(), static_cast<float>(i + 1));
        }
        auto* data1 = input1.data<float>();
        std::fill(data1, data1 + input1.get_size(), 100.f);
    }

    void set(ov::InferRequest& req) {
        req.set_tensors("tensor_input0", input0Tensors);
        req.set_tensor("tensor_input1", input1);
    }
};

const std::string OUTPUT_BATCH_MISMATCH = "while the batch size of the inputs is";

// Sets 5 samples as inputs and an output tensor with another batch size, in the given setter order. Outputs never
// decide the batch size, the inputs do: the inference must be rejected with a clear error (not run with fewer command
// lists, skipping samples), and it must run with all the samples once an output with the right batch size is set.
void checkMismatchedOutputIsRejectedThenFixed(ov::InferRequest& req, const bool outputFirst) {
    const Shape oneSampleShape{1, 2, 2, 2};
    SetterOrderInputs inputs(5, oneSampleShape);
    ov::Tensor smallerOutput(ov::element::f32, Shape{2, 2, 2, 2});

    if (outputFirst) {
        req.set_tensor("tensor_output0", smallerOutput);
        inputs.set(req);
    } else {
        inputs.set(req);
        req.set_tensor("tensor_output0", smallerOutput);
    }

    inputs.fill();
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr(OUTPUT_BATCH_MISMATCH));

    // The error is reported again, nothing was changed by the rejected inference
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr(OUTPUT_BATCH_MISMATCH));

    ov::Tensor matchingOutput(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_output0", matchingOutput);
    inputs.fill();
    req.infer();
    ASSERT_NO_FATAL_FAILURE(checkAllSamplesWereComputed(req, 5, oneSampleShape));
    // The result is written in the tensor set by the user
    ASSERT_EQ(req.get_tensor("tensor_output0").data(), matchingOutput.data());
}

}  // namespace

// Setter order: the inputs are set first, then an output tensor with a smaller batch
TEST_P(DynamicBatchedTensorsRunTests, SetterOrderInputsThenMismatchedOutputTensorIsRejected) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();
    ASSERT_NO_FATAL_FAILURE(checkMismatchedOutputIsRejectedThenFixed(req, false));
}

// Setter order: the output tensor is set before the inputs
TEST_P(DynamicBatchedTensorsRunTests, SetterOrderOutputTensorThenInputsMismatchIsRejected) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();
    ASSERT_NO_FATAL_FAILURE(checkMismatchedOutputIsRejectedThenFixed(req, true));
}

// The batch of the inputs grows after an inference, while the output set for the previous batch is kept
TEST_P(DynamicBatchedTensorsRunTests, SetterOrderBiggerInputsAfterInferenceKeepingSmallerOutputTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    const Shape oneSampleShape{1, 2, 2, 2};
    SetterOrderInputs smallInputs(2, oneSampleShape);
    ov::Tensor smallOutput(ov::element::f32, Shape{2, 2, 2, 2});
    smallInputs.set(req);
    req.set_tensor("tensor_output0", smallOutput);
    smallInputs.fill();
    req.infer();
    ASSERT_NO_FATAL_FAILURE(checkAllSamplesWereComputed(req, 2, oneSampleShape));

    SetterOrderInputs bigInputs(6, oneSampleShape);
    bigInputs.set(req);
    bigInputs.fill();
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr(OUTPUT_BATCH_MISMATCH));

    ov::Tensor bigOutput(ov::element::f32, Shape{6, 2, 2, 2});
    req.set_tensor("tensor_output0", bigOutput);
    bigInputs.fill();
    req.infer();
    ASSERT_NO_FATAL_FAILURE(checkAllSamplesWereComputed(req, 6, oneSampleShape));
}

// After a rejected inference (inputs with different batch sizes), the input that caused it is set back to the batch
// size of the other one with set_tensor: the request must recover and run with the original batch size
TEST_P(DynamicBatchedTensorsRunTests, RecoverAfterRejectedInferenceBySettingTensorBack) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    ov::Tensor changedInput0(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input0", changedInput0);
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    // Back to the batch size of the other input, with a different tensor...
    ov::Tensor revertedInput0(ov::element::f32, Shape{2, 2, 2, 2});
    OV_ASSERT_NO_THROW(req.set_tensor("tensor_input0", revertedInput0));
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, revertedInput0, input1, 2.f));

    // ... and the request keeps accepting batch changes afterwards
    ov::Tensor newInput0(ov::element::f32, Shape{4, 2, 2, 2});
    ov::Tensor newInput1(ov::element::f32, Shape{4, 2, 2, 2});
    req.set_tensor("tensor_input0", newInput0);
    req.set_tensor("tensor_input1", newInput1);
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, newInput0, newInput1, 3.f));
}

// Same, but the changed input is reshaped back (set_shape, without calling its setter) to the original batch size
TEST_P(DynamicBatchedTensorsRunTests, RecoverAfterRejectedInferenceByReshapingInputBack) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    ov::Tensor changedInput0(ov::element::f32, Shape{5, 2, 2, 2});
    req.set_tensor("tensor_input0", changedInput0);
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    changedInput0.set_shape(Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, changedInput0, input1, 2.f));
}

// The rejection can also come from a reshape: after reshaping only one input, reshaping it back recovers the request
TEST_P(DynamicBatchedTensorsRunTests, RecoverAfterRejectedReshapeByReshapingInputBack) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{2, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(setInputsAndInferOnce(req, input0, input1));

    input0.set_shape(Shape{5, 2, 2, 2});
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));

    input0.set_shape(Shape{2, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 2.f));

    // Both inputs can still be moved to a new batch afterwards
    input0.set_shape(Shape{6, 2, 2, 2});
    input1.set_shape(Shape{6, 2, 2, 2});
    ASSERT_NO_FATAL_FAILURE(fillInferAndCheckTwoInputs(req, input0, input1, 3.f));
}

// Both inputs get a tensor with a different batch size on a fresh request: reported when inferring
TEST_P(DynamicBatchedTensorsRunTests, SetTensorInputsWithDifferentBatchSizesFails) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto req = createTwoInputsDynamicBatchRequest();

    ov::Tensor input0(ov::element::f32, Shape{3, 2, 2, 2});
    ov::Tensor input1(ov::element::f32, Shape{6, 2, 2, 2});
    req.set_tensor("tensor_input0", input0);
    req.set_tensor("tensor_input1", input1);
    OV_EXPECT_THROW(req.infer(), ov::Exception, HasSubstr("Input tensors have different batch sizes"));
}

TEST_P(DynamicBatchedTensorsRunTests, DynamicSetInputDifferentTensorsMultipleInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
            ASSERT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

using BatchedTensorsRunFailureTests = BatchedTensorsRunTests;

TEST_P(BatchedTensorsRunFailureTests, FailedToDetectBatch) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto shape = Shape{4, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, shape, "");

    EXPECT_THROW(auto execNet = core->compile_model(model, target_device, configuration);, std::exception);
}

TEST_P(BatchedTensorsRunFailureTests, SetInputTensorsNotMatchingDynamicModelShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    size_t model_batch_upper_bound = 10;
    auto modelShape = PartialShape{ov::Dimension(1, model_batch_upper_bound), 2, 2, 2};
    auto model = BatchedTensorsRunTests::create_n_inputs(1, element::f32, modelShape, "N...");
    auto execNet = core->compile_model(model, target_device, configuration);
    ov::InferRequest req = execNet.create_infer_request();

    auto make_tensors = [](size_t count, const Shape& shape) {
        std::vector<ov::Tensor> tensors;
        for (size_t i = 0; i < count; ++i) {
            tensors.emplace_back(element::f32, shape);
        }
        return tensors;
    };

    // More tensors than the batch upper bound
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors(model_batch_upper_bound + 1, {1, 2, 2, 2})),
                 ov::Exception);
    // Batch within bounds but non-batch dimension not matching the model
    EXPECT_THROW(req.set_tensors("tensor_input0", make_tensors(4, {1, 2, 2, 3})), ov::Exception);

    OV_ASSERT_NO_THROW(req.set_tensors("tensor_input0", make_tensors(4, {1, 2, 2, 2})));
    OV_ASSERT_NO_THROW(req.infer());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> batchedConfigs = {{ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)},
                                                {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
                                                {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::AUTO)}};

const std::vector<ov::AnyMap> compatibilityBatchedConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         BatchedTensorsRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(batchedConfigs)),
                         BatchedTensorsRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         BatchedTensorsRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compatibilityBatchedConfigs)),
                         BatchedTensorsRunTests::getTestCaseName);

const std::vector<ov::AnyMap> PluginBatchedConfigs = {{ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DynamicBatchedTensorsRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(PluginBatchedConfigs)),
                         BatchedTensorsRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         BatchedTensorsRunFailureTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(PluginBatchedConfigs)),
                         BatchedTensorsRunTests::getTestCaseName);
