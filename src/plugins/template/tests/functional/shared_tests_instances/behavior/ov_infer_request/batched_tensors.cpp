// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/batched_tensors.hpp"

#include <vector>

namespace ov {
namespace test {
namespace behavior {

// Default implementation - exception is thrown when N is not first in layout
TEST_P(OVInferRequestBatchedTests, SetInputTensors_Batch_Non_0) {
    size_t batch = 3;
    auto one_shape = Shape{1, 3, 3, 3};
    auto batch_shape = Shape{batch, 3, 3, 3};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "CNHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch, ov::Tensor(element::f32, one_shape));
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

// Default implementation - exception is thrown when some tensor is remote
TEST_P(OVInferRequestBatchedTests, SetInputTensors_remote_tensor_default) {
    size_t batch = 4;
    auto one_shape = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 4, 4, 4};
    auto model = OVInferRequestBatchedTests::create_n_inputs(1, element::f32, batch_shape, "NCHW");
    const std::string tensor_name = "tensor_input0";
    auto execNet = ie->compile_model(model, target_device);
    ov::InferRequest req;
    req = execNet.create_infer_request();
    std::vector<ov::Tensor> tensors(batch - 1, ov::Tensor(element::f32, one_shape));
    tensors.emplace_back(ov::RemoteTensor());
    ASSERT_THROW(req.set_tensors(tensor_name, tensors), ov::Exception);
}

// Default implementation - throw on complicated ROI blobs combining
TEST_P(OVInferRequestBatchedTests, SetInputTensors_Strides) {
    size_t batch = 2;
    auto one_shape = Shape{1, 2, 2, 2};
    auto one_shape_stride = Shape{1, 4, 4, 4};
    auto batch_shape = Shape{batch, 2, 2, 2};
    auto one_shape_size_stride = ov::shape_size(one_shape_stride);
    auto model = OVInferRequestBatchedTests::create_n_inputs(2, element::f32, batch_shape, "NCHW");
    std::vector<float> buffer1(one_shape_size_stride, 10);
    std::vector<float> buffer2(one_shape_size_stride, 20);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    auto tensor1 = ov::Tensor(element::f32, one_shape_stride, &buffer1[0]);
    auto tensor2 = ov::Tensor(element::f32, one_shape_stride, &buffer2[0]);
    auto tensor1_cut = ov::Tensor(tensor1, {0, 1, 1, 1}, {1, 3, 3, 3});
    auto tensor2_cut = ov::Tensor(tensor1, {0, 1, 1, 1}, {1, 3, 3, 3});
    std::vector<ov::Tensor> tensors;
    tensors.push_back(tensor1_cut);
    tensors.push_back(tensor2_cut);
    auto exp_tensor = ov::Tensor(element::f32, batch_shape);
    ASSERT_THROW(
        {
            req.set_tensors("tensor_input0", tensors);
            req.infer();
        },
        ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

namespace {

using namespace ov::test::behavior;
using namespace ov;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestBatchedTests,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                         OVInferRequestBatchedTests::getTestCaseName);

}  // namespace
