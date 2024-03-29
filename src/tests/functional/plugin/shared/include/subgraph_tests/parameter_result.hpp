// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/parameter_result.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {

TEST_P(ParameterResultSubgraphTest, Inference) {
    run();
}

TEST_P(ParameterResultSubgraphTest, CheckSharedTensor) {
    ov::test::InputShape input_shape;
    std::tie(input_shape, targetDevice) = this->GetParam();

    ov::Shape shape = input_shape.second[0];
    auto input = ov::Tensor(ov::element::f32, shape);

    // Load model
    ov::Core core;
    auto compiled_model = core.compile_model(function, targetDevice);

    // Infer
    auto infer_req = compiled_model.create_infer_request();
    infer_req.set_input_tensor(input);
    infer_req.infer();

    ASSERT_EQ(infer_req.get_input_tensor().data(), infer_req.get_output_tensor().data());
}

}  // namespace test
}  // namespace ov
