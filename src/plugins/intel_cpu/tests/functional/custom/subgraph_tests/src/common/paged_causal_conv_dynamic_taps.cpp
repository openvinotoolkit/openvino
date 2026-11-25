// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

ov::Tensor ints_tensor(const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, {values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    return tensor;
}

TEST(PagedCausalConv1DDynamicTaps, PreservesPackedHistoryAndReuse) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    auto tokens = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 2});
    tokens->output(0).set_names({"tokens"});
    auto expanded = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{tokens, tokens, tokens, tokens}, 1);
    auto table = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 8, 3});
    table->output(0).set_names({"conv_state_table.0"});
    auto weights =
        Constant::create(ov::element::f32, {8, 1, 3}, {1.f, 2.f, 3.f, -1.f, 0.5f, 2.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f,
                                                       0.f, 1.f, 0.f, 0.f,  1.f,  0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f});
    auto bias = Constant::create(ov::element::f32, {0}, std::vector<float>{});
    const auto parameter = [](const char* name) {
        auto value = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{-1});
        value->output(0).set_names({name});
        return value;
    };
    auto subsequence_begins = parameter("subsequence_begins");
    auto block_indices = parameter("la.block_indices");
    auto block_indices_begins = parameter("la.block_indices_begins");
    auto past_lens = parameter("la.past_lens");
    auto cache_interval = parameter("la.cache_interval");
    auto paged = std::make_shared<ov::op::internal::PagedCausalConv1D>(expanded,
                                                                       table,
                                                                       weights,
                                                                       bias,
                                                                       subsequence_begins,
                                                                       block_indices,
                                                                       block_indices_begins,
                                                                       past_lens,
                                                                       cache_interval,
                                                                       1);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{paged},
                                             ov::ParameterVector{tokens,
                                                                 table,
                                                                 subsequence_begins,
                                                                 block_indices,
                                                                 block_indices_begins,
                                                                 past_lens,
                                                                 cache_interval});
    auto compiled = ov::Core().compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    ov::Tensor state(ov::element::f32, {4, 8, 3});
    std::fill_n(state.data<float>(), state.get_size(), 0.f);
    request.set_tensor("conv_state_table.0", state);

    const auto infer = [&](const std::vector<float>& values,
                           const std::vector<int32_t>& begins,
                           const std::vector<int32_t>& indices,
                           const std::vector<int32_t>& past,
                           const std::vector<std::vector<float>>& expected) {
        ov::Tensor token_tensor(ov::element::f32, {values.size() / 2, 2});
        std::copy(values.begin(), values.end(), token_tensor.data<float>());
        request.set_tensor("tokens", token_tensor);
        request.set_tensor("subsequence_begins", ints_tensor(begins));
        request.set_tensor("la.block_indices", ints_tensor(indices));
        request.set_tensor("la.block_indices_begins", ints_tensor({0, 2, 4}));
        request.set_tensor("la.past_lens", ints_tensor(past));
        request.set_tensor("la.cache_interval", ints_tensor({0, 0}));
        request.infer();
        const auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_size(), values.size() * expected.size());
        for (size_t output = 0; output < expected.size(); ++output) {
            ASSERT_EQ(expected[output].size(), values.size());
            for (size_t i = 0; i < values.size(); ++i)
                EXPECT_FLOAT_EQ(actual.data<float>()[(i / 2) * 8 + output * 2 + i % 2], expected[output][i])
                    << "output=" << output << " index=" << i;
        }
    };
    infer({1, 10, 2, 20, 3, 30, 4, 40, 5, 50},
          {0, 3, 5},
          {0, 1, 2, 3},
          {0, 0},
          {{3, 20, 8, 45, 14, 60, 12, 80, 23, 120},
           {1, 10, 2, 20, 3, 30, 4, 40, 5, 50},
           {0, 0, 1, 10, 2, 20, 0, 0, 4, 40},
           {0, 0, 0, 0, 1, 10, 0, 0, 0, 0}});
    infer({6, 60, 7, 70},
          {0, 1, 2},
          {1, 0, 3, 2},
          {3, 2},
          {{26, 115, 35, 125}, {6, 60, 7, 70}, {3, 30, 5, 50}, {2, 20, 4, 40}});
}

}  // namespace
