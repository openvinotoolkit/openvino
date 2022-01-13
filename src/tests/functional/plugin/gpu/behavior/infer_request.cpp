// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/core.hpp"

#include <common_test_utils/test_common.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"

using namespace ::testing;

TEST(TensorTest, smoke_canSetShapeForPreallocatedTensor) {
    auto ie = ov::runtime::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ngraph::builder::subgraph::makeSplitMultiConvConcat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    auto exec_net = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    // Check set_shape call for pre-allocated input/output tensors
    auto input_tensor = inf_req.get_input_tensor(0);
    ASSERT_NO_THROW(input_tensor.set_shape({1, 4, 20, 20}));
    ASSERT_NO_THROW(input_tensor.set_shape({1, 3, 20, 20}));
    ASSERT_NO_THROW(input_tensor.set_shape({2, 3, 20, 20}));
    auto output_tensor = inf_req.get_output_tensor(0);
    ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 12, 12}));
    ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 10, 10}));
    ASSERT_NO_THROW(output_tensor.set_shape({2, 10, 20, 20}));
}
