// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/opsets/opset8.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "behavior/ov_infer_request/batched_tensors.hpp"
#include "common_test_utils/file_utils.hpp"
#include <chrono>

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestBatchedTests::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    return "target_device=" + obj.param;
}

std::string OVInferRequestBatchedTests::generateCacheDirName(const std::string& test_name) {
    using namespace std::chrono;
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto hash = std::to_string(std::hash<std::string>()(test_name));
    std::stringstream ss;
    auto ts = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch());
    ss << hash << "_" << "_" << ts.count();
    return ss.str();
}

void OVInferRequestBatchedTests::SetUp() {
    target_device = GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    m_cache_dir = generateCacheDirName(GetTestName());
}

void OVInferRequestBatchedTests::TearDown() {
    if (m_need_reset_core) {
        ie->set_property({ov::cache_dir()});
        ie.reset();
        ov::test::utils::PluginCache::get().reset();
        ov::test::utils::removeFilesWithExt(m_cache_dir, "blob");
        ov::test::utils::removeDir(m_cache_dir);
    }
    APIBaseTest::TearDown();
}

std::shared_ptr<Model> OVInferRequestBatchedTests::create_n_inputs(size_t n, element::Type type,
                                              const PartialShape& shape, const ov::Layout& layout) {
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

TEST_P(OVInferRequestBatchedTests, SetInputTensorsBase) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    auto exp_tensor = ov::Tensor(element::f32, batch_shape);
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);
    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        auto *f = tensors[i].data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = 5.f;
        }
    }
    req.infer(); // Adds '1' to each element
    for (auto j = 0; j < one_shape_size * batch; ++j) {
        EXPECT_NEAR(actual[j], 6.f, 1e-5) << "Expected=6, actual=" << actual[j] << " for index " << j;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensorsAsync) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    auto exp_tensor = ov::Tensor(element::f32, batch_shape);
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);
    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        auto *f = tensors[i].data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = 5.f;
        }
    }
    req.start_async(); // Adds '1' to each element
    req.wait_for(std::chrono::milliseconds(1000));
    for (auto j = 0; j < one_shape_size * batch; ++j) {
        EXPECT_NEAR(actual[j], 6.f, 1e-5) << "Expected=6, actual=" << actual[j] << " for index " << j;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_override_with_set) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    std::vector<float> buffer(one_shape_size * batch, 4);
    std::vector<float> buffer2(one_shape_size * batch, 5);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    auto exp_tensor = ov::Tensor(element::f32, batch_shape);
    for (auto i = 0; i < batch; ++i) {
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[i * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    auto plain_tensor = ov::Tensor(element::f32, batch_shape, &buffer2[0]);
    req.set_tensors("tensor_input0", tensors);
    req.set_tensor("tensor_input0", plain_tensor);
    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto i = 0; i < batch; ++i) {
        auto *f = tensors[i].data<float>();
        for (auto j = 0; j < one_shape_size; ++j) {
            f[j] = 5.f;
            buffer2[j + i * one_shape_size] = 55.f;
        }
    }
    req.infer(); // Adds '1' to each element
    for (auto j = 0; j < one_shape_size * batch; ++j) {
        EXPECT_NEAR(actual[j], 56.f, 1e-5) << "Expected=56, actual=" << actual[j] << " for index " << j;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensorsBase_Caching) {
    m_need_reset_core = true;
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "N...");
    ie->set_property({ov::cache_dir(m_cache_dir)});
    auto execNet_no_cache = ie->compile_model(model, target_device);
    auto execNet_cache = ie->compile_model(model, target_device);
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);

    // Create InferRequest
    ov::InferRequest req;
    req = execNet_cache.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_input_tensors(tensors);
    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 5; testNum++) {
        for (auto i = 0; i < batch; ++i) {
            auto *f = tensors[i].data<float>();
            for (auto j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 10);
            }
        }
        req.infer(); // Adds '1' to each element
        for (auto j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 11) << "Infer " << testNum << ": Expected=" << testNum + 11
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Multiple_Infer) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (auto i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    auto actual_tensor = req.get_tensor("tensor_output0");
    auto* actual = actual_tensor.data<float>();
    for (auto testNum = 0; testNum < 5; testNum++) {
        for (auto i = 0; i < batch; ++i) {
            auto *f = tensors[i].data<float>();
            for (auto j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + 20);
            }
        }
        req.infer(); // Adds '1' to each element
        for (auto j = 0; j < one_shape_size * batch; ++j) {
            EXPECT_EQ(actual[j], testNum + 21) << "Infer " << testNum << ": Expected=" << testNum + 21
                                               << ", actual=" << actual[j] << " for index " << j;
        }
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Can_Infer_Dynamic) {
    size_t batch = 4;
    auto one_shape = Shape{1, 2, 2, 2};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, PartialShape({-1, 2, 2, 2}), "N...");
    // Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    std::vector<float> buffer(one_shape_size * batch * 2, 0);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    for (size_t i = 0; i < batch; ++i) {
        // non contiguous memory (i*2)
        auto tensor = ov::Tensor(element::f32, one_shape, &buffer[(i * 2) * one_shape_size]);
        tensors.push_back(std::move(tensor));
    }
    req.set_tensors("tensor_input0", tensors);

    for (size_t testNum = 0; testNum < 2; testNum++) {
        for (size_t i = 0; i < batch; ++i) {
            auto *f = tensors[i].data<float>();
            for (size_t j = 0; j < one_shape_size; ++j) {
                f[j] = static_cast<float>(testNum + i);
            }
        }
        req.infer(); // Adds '1' to each element
        auto actual_tensor = req.get_tensor("tensor_output0");
        auto* actual = actual_tensor.data<float>();
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < one_shape_size; ++j) {
                EXPECT_EQ(actual[j + i * one_shape_size], testNum + i + 1)
                                    << "Infer " << testNum << ": Expected=" << testNum + i + 1
                                    << ", actual=" << actual[ + i * one_shape_size] << " for index " << j;
            }
        }
    }
}

TEST_P(OVInferRequestBatchedTests, SetTensors_Batch1) {
    auto one_shape = Shape{1, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, one_shape, "N...");
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    auto exp_tensor = ov::Tensor(element::f32, one_shape);
    auto* exp = exp_tensor.data<float>();
    auto tensor = ov::Tensor(element::f32, one_shape);
    auto* f = tensor.data<float>();
    for (size_t j = 0; j < one_shape_size; ++j) {
        f[j] = static_cast<float>(j);
        exp[j] = f[j];
    }
    tensors.push_back(std::move(tensor));
    const std::string tensor_name = "tensor_input0";
    req.set_tensors(tensor_name, tensors);
    auto actual_tensor = req.get_tensor(tensor_name);
    ASSERT_EQ(exp_tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(exp_tensor.get_element_type(), actual_tensor.get_element_type());
    auto* actual = actual_tensor.data<float>();
    for (size_t i = 0; i < one_shape_size; ++i) {
        EXPECT_EQ(exp[i], actual[i]) << "Expected=" << exp[i] << ", actual=" << actual[i] << " for " << i;
    }
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Get_Tensor_Not_Allowed) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    req.set_tensors(tensor_name, tensors);
    ASSERT_THROW(req.get_tensor(tensor_name), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_No_Batch) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "DCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "undefined";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "undefined";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetTensors_Friendly_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_No_index) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(10, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_no_name_multiple_inputs) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_input_tensors(tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_count) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch + 1, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Empty_Array) {
    size_t batch = 3;
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_diff_batches) {
    auto batch_shape = Shape{3, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    tensors.emplace_back(element::f32, Shape{2, 3, 3, 3});
    tensors.emplace_back(element::f32, Shape{1, 3, 3, 3});
    // This expectation can be updated if non-equal sizes of tensors is supported in future
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Correct_all) {
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{2, 3, 3, 3};
    std::vector<float> buffer(ov::shape_size(batch_shape), 1);
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors;
    tensors.emplace_back(element::f32, one_shape, buffer.data());
    tensors.emplace_back(element::f32, one_shape, buffer.data() + ov::shape_size(one_shape));
    OV_ASSERT_NO_THROW(req.set_input_tensors(tensors));
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Cache_CheckDeepCopy) {
    m_need_reset_core = true;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{2, 3, 3, 3};
    std::vector<float> buffer(ov::shape_size(batch_shape), 1);
    std::vector<float> buffer_out(ov::shape_size(batch_shape), 1);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "NCHW");
    ie->set_property({ov::cache_dir(m_cache_dir)});
    auto execNet_no_cache = ie->compile_model(model, target_device);
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    model->input(0).set_names({"updated_input0"}); // Change param name of original model
    model->get_parameters()[0]->set_layout("????");
    model->output(0).set_names({"updated_output0"}); // Change result name of original model
    std::vector<ov::Tensor> tensors;
    tensors.emplace_back(element::f32, one_shape, buffer.data());
    tensors.emplace_back(element::f32, one_shape, buffer.data() + ov::shape_size(one_shape));
    auto out_tensor = ov::Tensor(element::f32, batch_shape, buffer_out.data());
    // Verify that infer request still has its own copy of input/output, user can use old names
    OV_ASSERT_NO_THROW(req.set_tensors("tensor_input0", tensors));
    OV_ASSERT_NO_THROW(req.set_tensor("tensor_output0", out_tensor));
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_tensor_element_type) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch - 1, ov::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::u8, one_shape);
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_tensor_shape) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch - 1, ov::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::f32, Shape{1, 4, 2, 8});
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}


}  // namespace behavior
}  // namespace test
}  // namespace ov
