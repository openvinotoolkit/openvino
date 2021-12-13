// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"

using namespace ov::test::behavior;
using namespace ov;

namespace {

TEST_P(OVInferRequestBatchedTests, SetTensors_Batch1) {
    auto one_shape = Shape{1, 3, 10, 10};
    auto one_shape_size = ov::shape_size(one_shape);
    auto model = create_n_inputs(1, element::f32, one_shape, "N...");
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
    const std::string tensor_name = "tensor_input0";
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
    auto model = create_n_inputs(1, element::f32, batch_shape, "DCHWN");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_Non_0) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "CNHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_No_Batch) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "DCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "undefined";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetTensors_No_Name) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
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
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
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
    auto model = create_n_inputs(2, element::f32, batch_shape, "NCHW");
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
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch + 1, runtime::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Empty_Array) {
    size_t batch = 3;
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_diff_batches) {
    auto batch_shape = Shape{3, 3, 3, 3};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors;
    tensors.emplace_back(element::f32, Shape{2, 3, 3, 3});
    tensors.emplace_back(element::f32, Shape{1, 3, 3, 3});
    // This expectation can be updated if non-equal sizes of tensors is supported in future
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Correct_all) {
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{2, 3, 3, 3};
    std::vector<float> buffer(ov::shape_size(batch_shape), 1);
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
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
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::u8, one_shape);
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_Incorrect_tensor_shape) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(element::f32, Shape{1, 4, 2, 8});
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

TEST_P(OVInferRequestBatchedTests, SetInputTensors_remote_tensor_default) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, targetDevice);
    ov::runtime::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::runtime::Tensor> tensors(batch - 1, runtime::Tensor(element::f32, one_shape));
    tensors.emplace_back(runtime::RemoteTensor());
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CPU, OVInferRequestBatchedTests,
                         ::testing::Values(CommonTestUtils::DEVICE_CPU),
                         OVInferRequestBatchedTests::getTestCaseName);

}  // namespace
