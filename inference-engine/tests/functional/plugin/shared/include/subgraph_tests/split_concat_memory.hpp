// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_concat_memory.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConcatMemory, cyclicBufferCorrectness) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto ie = PluginCache::get().ie();
    cnnNetwork = InferenceEngine::CNNNetwork{function};

    auto exe_net = ie->LoadNetwork(cnnNetwork, "CPU");
    auto inf_reg = exe_net.CreateInferRequest();

    /*
     * cnc1 out  |  mem      | In|q
     *           |===============|
     * iter_1    | 0 | 0 | 0 | 1 |
     * iter_2    | 0 | 0 | 1 | 2 |
     * iter 3    | 0 | 1 | 2 | 3 |
     */

    auto i_blob = inf_reg.GetBlob("input");
    auto o_blob = inf_reg.GetBlob("plus_one");

    auto o_blob_ref = make_blob_with_precision(o_blob->getTensorDesc());
    o_blob_ref->allocate();

    auto fill_by_quarter = [this] (InferenceEngine::Blob::Ptr& blob, std::vector<float> vals) {
        IE_ASSERT(vals.size() == 4);
        auto quarter_blocked_shape = blob->getTensorDesc().getDims();

        // splis axis dimension into chunk
        IE_ASSERT(quarter_blocked_shape[axis] % vals.size() == 0);
        quarter_blocked_shape[axis] /= vals.size();
        quarter_blocked_shape.insert(quarter_blocked_shape.begin() + axis, vals.size());

        auto quarter_blocked_view = CommonTestUtils::make_reshape_view(blob, quarter_blocked_shape);
        CommonTestUtils::fill_data_with_broadcast(quarter_blocked_view, axis, vals);
    };

    // iteration 1
    CommonTestUtils::fill_data_const(i_blob, 1);
    fill_by_quarter(o_blob_ref, {1, 1, 1, 2});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 2
    CommonTestUtils::fill_data_const(i_blob, 2);
    fill_by_quarter(o_blob_ref, {1, 1, 2, 3});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 3
    CommonTestUtils::fill_data_const(i_blob, 3);
    fill_by_quarter(o_blob_ref, {1, 2, 3, 4});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);
}

}  // namespace SubgraphTestsDefinitions