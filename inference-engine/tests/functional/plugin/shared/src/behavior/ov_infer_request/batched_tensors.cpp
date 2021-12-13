// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/opsets/opset8.hpp>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "behavior/ov_infer_request/batched_tensors.hpp"
#include "common_test_utils/file_utils.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono;

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestBatchedTests::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    return "targetDevice=" + obj.param;
}

std::string OVInferRequestBatchedTests::generateCacheDirName(const std::string& test_name) {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto hash = std::to_string(std::hash<std::string>()(test_name));
    std::stringstream ss;
    auto ts = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
    ss << hash << "_" << std::this_thread::get_id() << "_" << ts.count();
    return ss.str();
}

void OVInferRequestBatchedTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    targetDevice = GetParam();
    m_cache_dir = generateCacheDirName(GetTestName());
}

void OVInferRequestBatchedTests::TearDown() {
    CommonTestUtils::removeFilesWithExt(m_cache_dir, "blob");
    CommonTestUtils::removeDir(m_cache_dir);
}

namespace {
template <int N>
static std::shared_ptr<Model> create_n_inputs(element::Type type,
                                              const PartialShape& shape,
                                              const ov::Layout& layout) {
    ResultVector res;
    ParameterVector params;
    for (size_t i = 0; i < N; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<opset8::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str, "input" + index_str});
        data1->set_layout(layout);
        auto constant = opset8::Constant::create(type, {1}, {1});
        auto op1 = std::make_shared<opset8::Add>(data1, constant);
        op1->set_friendly_name("Add" + index_str);
        auto res1 = std::make_shared<opset8::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str, "Result" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    return std::make_shared<Model>(res, params);
}

} //namespace

TEST_P(OVInferRequestBatchedTests, SetInputTensorsBase) {
    size_t batch = 4;
    auto one_shape = Shape{1, 3, 10, 10};
    auto batch_shape = Shape{batch, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs<1>(element::f32, batch_shape, "N...");
    auto execNet = ie->compile_model(model, targetDevice);
    // Create InferRequest
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    auto exp_tensor = ov::runtime::Tensor(element::f32, batch_shape);
    auto* exp = exp_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        auto tensor = runtime::Tensor(element::f32, one_shape);
        auto* f = tensor.data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = static_cast<float>(j + i);
            exp[one_shape_size * i + j] = f[j];
        }
        tensors.push_back(std::move(tensor));
    }
    req.set_input_tensors(tensors);
    auto actual_tensor = req.get_input_tensor();
    ASSERT_EQ(exp_tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(exp_tensor.get_element_type(), actual_tensor.get_element_type());
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < one_shape_size * batch; ++i) {
        EXPECT_EQ(exp[i], actual[i]) << "Expected=" << exp[i] << ", actual=" << actual[i] << " for index " << i;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensorsBase_Caching) {
    size_t batch = 4;
    auto one_shape = Shape{1, 3, 10, 10};
    auto batch_shape = Shape{batch, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs<1>(element::f32, batch_shape, "N...");
    ie->set_config({{CONFIG_KEY(CACHE_DIR), m_cache_dir}});
    auto execNet_no_cache = ie->compile_model(model, targetDevice);
    auto execNet_cache = ie->compile_model(model, targetDevice);

    // Create InferRequest
    ov::runtime::InferRequest req;
    req = execNet_cache.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    auto exp_tensor = ov::runtime::Tensor(element::f32, batch_shape);
    auto* exp = exp_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        auto tensor = runtime::Tensor(element::f32, one_shape);
        auto* f = tensor.data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = static_cast<float>(j + i);
            exp[one_shape_size * i + j] = f[j];
        }
        tensors.push_back(std::move(tensor));
    }
    req.set_input_tensors(tensors);
    auto actual_tensor = req.get_input_tensor();
    ASSERT_EQ(exp_tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(exp_tensor.get_element_type(), actual_tensor.get_element_type());
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < one_shape_size * batch; ++i) {
        EXPECT_EQ(exp[i], actual[i]) << "Expected=" << exp[i] << ", actual=" << actual[i] << " for index " << i;
    }
    PluginCache::get().reset();
}

TEST_P(OVInferRequestBatchedTests, SetTensors_Batch1) {
    auto one_shape = Shape{1, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs<1>(element::f32, one_shape, "N...");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    // Create InferRequest
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    auto exp_tensor = ov::runtime::Tensor(element::f32, one_shape);
    auto* exp = exp_tensor.data<float>();
    auto tensor = runtime::Tensor(element::f32, one_shape);
    auto* f = tensor.data<float>();
    for (auto j = 0; j < one_shape_size; ++j) {
        f[j] = static_cast<float>(j);
        exp[j] = f[j];
    }
    tensors.push_back(std::move(tensor));
    req.set_tensors(tensor_name, tensors);
    auto actual_tensor = req.get_tensor(tensor_name);
    ASSERT_EQ(exp_tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(exp_tensor.get_element_type(), actual_tensor.get_element_type());
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < one_shape_size; ++i) {
        EXPECT_EQ(exp[i], actual[i]) << "Expected=" << exp[i] << ", actual=" << actual[i] << " for " << i;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_Incorrect) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "DCHWN");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_Non_0) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "CNHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_No_Batch) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "DCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "undefined";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "undefined";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_No_index) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(10, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_no_name_multiple_inputs) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<2>(element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_count) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch + 1, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Empty_Array) {
    size_t batch = 3;
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_diff_batches) {
    auto batch_shape = Shape{3, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    tensors.emplace_back(element::f32, Shape{2, 3, 3, 3});
    tensors.emplace_back(element::f32, Shape{1, 3, 3, 3});
    // This expectation can be updated if non-equal sizes of tensors is supported in future
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Correct_all) {
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{2, 3, 3, 3};
    std::vector<float> buffer(ov::shape_size(batch_shape), 1);
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    tensors.emplace_back(element::f32, one_shape, buffer.data());
    tensors.emplace_back(element::f32, one_shape, buffer.data() + ov::shape_size(one_shape));
    ASSERT_NO_THROW(req.set_input_tensors(tensors));
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_tensor_element_type) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::u8, one_shape);
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_tensor_shape) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::f32, Shape{1, 4, 2, 8});
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_remote_tensor_default) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = create_n_inputs<1>(element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(runtime::RemoteTensor());
    ASSERT_THROW(req.set_input_tensors(tensor_name, tensors), ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
