// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/data_utils.hpp"
#include "shared_test_classes/subgraph/split_concat_memory.hpp"

namespace ov {
namespace test {

TEST_P(SplitConcatMemory, cyclicBufferCorrectness) {
    /*
     * cnc1 out  |  mem      | In|q
     *           |===============|
     * iter_1    | 0 | 0 | 0 | 1 |
     * iter_2    | 0 | 0 | 1 | 2 |
     * iter 3    | 0 | 1 | 2 | 3 |
     */

    compile_model();
    inferRequest = compiledModel.create_infer_request();

    auto i_tensor = inferRequest.get_tensor(*function->inputs().begin());

    auto o_tensor = inferRequest.get_tensor(*function->outputs().begin());
    auto output_tensor_ref = ov::Tensor(o_tensor.get_element_type(), o_tensor.get_shape());

    auto fill_by_quarter = [this](ov::Tensor& tensor, std::vector<float> vals) {
        OPENVINO_ASSERT(vals.size() == 4);
        auto quarter_blocked_shape = tensor.get_shape();

        // splis axis dimension into chunk
        OPENVINO_ASSERT(quarter_blocked_shape[axis] % vals.size() == 0);
        quarter_blocked_shape[axis] /= vals.size();
        quarter_blocked_shape.insert(quarter_blocked_shape.begin() + axis, vals.size());

        OPENVINO_ASSERT(ov::shape_size(quarter_blocked_shape) == tensor.get_size());
        auto quarter_blocked_view = ov::Tensor(tensor.get_element_type(), quarter_blocked_shape, tensor.data());

        ov::test::utils::fill_data_with_broadcast(quarter_blocked_view, axis, vals);
    };

    // iteration 1

    ov::test::utils::fill_data_with_broadcast(i_tensor, 0, {1});
    fill_by_quarter(output_tensor_ref, {1, 1, 1, 2});
    inferRequest.infer();
    compare({output_tensor_ref}, {o_tensor});

    // iteration 2
    ov::test::utils::fill_data_with_broadcast(i_tensor, 0, {2});
    fill_by_quarter(output_tensor_ref, {1, 1, 2, 3});
    inferRequest.infer();
    compare({output_tensor_ref}, {o_tensor});

    // iteration 3
    ov::test::utils::fill_data_with_broadcast(i_tensor, 0, {3});
    fill_by_quarter(output_tensor_ref, {1, 2, 3, 4});
    inferRequest.infer();
    compare({output_tensor_ref}, {o_tensor});
}

}  // namespace test
}  // namespace ov

