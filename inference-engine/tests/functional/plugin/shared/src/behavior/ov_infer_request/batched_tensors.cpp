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

std::shared_ptr<Model> OVInferRequestBatchedTests::create_n_inputs(size_t n, element::Type type,
                                              const PartialShape& shape, const ov::Layout& layout) {
    ResultVector res;
    ParameterVector params;
    for (size_t i = 0; i < n; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<opset8::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        data1->set_layout(layout);
        auto constant = opset8::Constant::create(type, {1}, {1});
        auto op1 = std::make_shared<opset8::Add>(data1, constant);
        op1->set_friendly_name("Add" + index_str);
        auto res1 = std::make_shared<opset8::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    return std::make_shared<Model>(res, params);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensorsBase) {
    size_t batch = 4;
    auto one_shape = Shape{1, 3, 10, 10};
    auto batch_shape = Shape{batch, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs(2, element::f32, batch_shape, "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, targetDevice);
    // Create InferRequest
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    auto exp_tensor = ov::runtime::Tensor(element::f32, batch_shape);
    auto* exp = exp_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = runtime::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        auto* f = tensor.data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = static_cast<float>(j + i);
            exp[one_shape_size * i + j] = f[j];
        }
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);
    auto actual_tensor = req.get_tensor("tensor_input0");
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
    auto model = create_n_inputs(1, element::f32, batch_shape, "N...");
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

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Multiple_Infer) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs(2, element::f32, batch_shape, "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, targetDevice);
    // Create InferRequest
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = runtime::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 10; testNum++) {
        for (auto i = 0; i < batch; ++i) {
            auto *f = tensors[i].data<float>();
            for (auto j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum);
            }
        }
        req.infer(); // Adds '1' to each element
        for (auto j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 1) << "Infer " << testNum << ": Expected=" << testNum + 1
                                              << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
