// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"

using namespace ov::test::behavior;
using namespace ov;

namespace {

// Default implementation - exception is thrown when N is not first in layout
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

// Default implementation - exception is thrown when some tensor is remote
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

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_TEMPLATE, OVInferRequestBatchedTests,
                         ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                         OVInferRequestBatchedTests::getTestCaseName);

}  // namespace
