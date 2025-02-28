// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/paged_attention.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "random"

namespace {

// 0: reference_tests::Tensor            -- query
// 1: reference_tests::Tensor            -- key
// 2: reference_tests::Tensor            -- value
// 3: reference_tests::Tensor            -- key_cache
// 4: reference_tests::Tensor            -- value_cache
// 5: reference_tests::Tensor            -- past_lens (shape: [batch_size_in_sequences])
// 6: reference_tests::Tensor            -- subsequence_begins (shape: [batch_size_in_sequences + 1])
// 7: reference_tests::Tensor            -- block_indices (shape: [num_blocks])
// 8: reference_tests::Tensor            -- block_indices_begins (shape: [batch_size_in_sequences + 1])
// 9: reference_tests::Tensor            -- scale
// 10: reference_tests::Tensor           -- sliding_window
// 11: reference_tests::Tensor           -- alibi_slopes (shape: [num_kv_heads])
// 12: reference_tests::Tensor           -- max_context_len
// 13: reference_tests::Tensor           -- rotated_block_indices (shape: [num_rotated_blocks])
// 14: reference_tests::Tensor           -- rotation_deltas (shape: as specified, e.g. [num_rotated_blocks, 1])
// 15: reference_tests::Tensor           -- rotation_trig_lut (shape: [lut_rows, head_size])
// 16: reference_tests::Tensor           -- output data to compare reference output with
// 17: std::string          -- targetDevice
using PagedAttentionParams = std::tuple<reference_tests::Tensor,  // 0: query
                                        reference_tests::Tensor,  // 1: key
                                        reference_tests::Tensor,  // 2: value
                                        reference_tests::Tensor,  // 3: key_cache
                                        reference_tests::Tensor,  // 4: value_cache
                                        reference_tests::Tensor,  // 5: past_lens
                                        reference_tests::Tensor,  // 6: subsequence_begins
                                        reference_tests::Tensor,  // 7: block_indices
                                        reference_tests::Tensor,  // 8: block_indices_begins
                                        reference_tests::Tensor,  // 9: scale
                                        reference_tests::Tensor,  // 10: sliding_window
                                        reference_tests::Tensor,  // 11: alibi_slopes
                                        reference_tests::Tensor,  // 12: max_context_len
                                        reference_tests::Tensor,  // 13: rotated_block_indices
                                        reference_tests::Tensor,  // 14: rotation_deltas
                                        reference_tests::Tensor,  // 15: rotation_trig_lut
                                        reference_tests::Tensor,  // 16: output data
                                        std::string>;             // 17: targetDevice
class ReferencePagedAttention : public testing::TestWithParam<PagedAttentionParams>,
                                public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        query = std::get<0>(params).data;
        key = std::get<1>(params).data;
        value = std::get<2>(params).data;
        key_cache = std::get<3>(params).data;
        value_cache = std::get<4>(params).data;
        past_lens = std::get<5>(params).data;
        subsequence_begins = std::get<6>(params).data;
        block_indices = std::get<7>(params).data;
        block_indices_begins = std::get<8>(params).data;
        scale = std::get<9>(params).data;
        sliding_window = std::get<10>(params).data;
        alibi_slopes = std::get<11>(params).data;
        max_context_len = std::get<12>(params).data;
        rotated_block_indices = std::get<13>(params).data;
        rotation_deltas = std::get<14>(params).data;
        rotation_trig_lut = std::get<15>(params).data;
        targetDevice = std::get<17>(params).data;

        function = CreateFunction(params);
        inputData = {query,
                     key,
                     value,
                     key_cache,
                     value_cache,
                     past_lens,
                     subsequence_begins,
                     block_indices,
                     block_indices_begins,
                     scale,
                     sliding_window,
                     alibi_slopes,
                     max_context_len,
                     rotated_block_indices,
                     rotation_deltas,
                     rotation_trig_lut};

        refOutData = {std::get<16>(params)};
    }

    static std::string tensor2str(const reference_tests::Tensor& t) {
        std::ostringstream oss;
        oss << "[type: ";
        oss << t.type;
        oss << ", shape: ";
        oss << t.shape.to_string();
        oss << "]";
        return oss.str();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttentionParams>& obj) {
        std::ostringstream name;

        name << "q=" << tensor2str(std::get<0>(obj.param)) << "_";
        name << "k=" << tensor2str(std::get<1>(obj.param)) << "_";
        name << "v=" << tensor2str(std::get<2>(obj.param)) << "_";
        name << "k_cache=" << tensor2str(std::get<3>(obj.param)) << "_";
        name << "v_cache=" << tensor2str(std::get<4>(obj.param)) << "_";
        name << "past_lens=" << tensor2str(std::get<5>(obj.param)) << "_";
        name << "subsequence_begins=" << tensor2str(std::get<6>(obj.param)) << "_";
        name << "block_indices=" << tensor2str(std::get<7>(obj.param)) << "_";
        name << "block_indices_begins=" << tensor2str(std::get<8>(obj.param)) << "_";
        name << "scale=" << tensor2str(std::get<9>(obj.param)) << "_";
        name << "sliding_window=" << tensor2str(std::get<10>(obj.param)) << "_";
        name << "alibi_slopes=" << tensor2str(std::get<11>(obj.param)) << "_";
        name << "max_context_len=" << tensor2str(std::get<12>(obj.param)) << "_";
        name << "rotated_block_indices=" << tensor2str(std::get<13>(obj.param)) << "_";
        name << "rotation_deltas=" << tensor2str(std::get<14>(obj.param)) << "_";
        name << "rotation_trig_lut=" << tensor2str(std::get<15>(obj.param)) << "_";
        name << "trgDev=" << std::get<17>(obj.param) << "_";

        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const PagedAttentionParams& params) {
        const auto& query = std::get<0>(params).data;
        const auto& key = std::get<1>(params).data;
        const auto& value = std::get<2>(params).data;
        const auto& key_cache = std::get<3>(params).data;
        const auto& value_cache = std::get<4>(params).data;
        const auto& scale = std::get<9>(params).data;
        const auto& sliding_window = std::get<10>(params).data;
        const auto& max_context_len = std::get<12>(params).data;
        const auto& past_lens = std::get<5>(params).data;
        const auto& subsequence_begins = std::get<6>(params).data;
        const auto& block_indices = std::get<7>(params).data;
        const auto& block_indices_begins = std::get<8>(params).data;
        const auto& alibi_slopes = std::get<11>(params).data;
        const auto& rotated_block_indices = std::get<13>(params).data;
        const auto& rotation_deltas = std::get<14>(params).data;
        const auto& rotation_trig_lut = std::get<15>(params).data;

        const std::vector<ov::Tensor> funcInputs = {query,
                                                    key,
                                                    value,
                                                    key_cache,
                                                    value_cache,
                                                    past_lens,
                                                    subsequence_begins,
                                                    block_indices,
                                                    block_indices_begins,
                                                    scale,
                                                    sliding_window,
                                                    alibi_slopes,
                                                    max_context_len,
                                                    rotated_block_indices,
                                                    rotation_deltas,
                                                    rotation_trig_lut};
        ov::ParameterVector inputParams;
        for (auto& input : funcInputs) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_shape()));
        }

        auto paged_attn = std::make_shared<ov::op::v16::PagedAttention>();
        return std::make_shared<ov::Model>(paged_attn->outputs(), inputParams);
    }

    ov::Tensor query;
    ov::Tensor key;
    ov::Tensor value;
    ov::Tensor key_cache;
    ov::Tensor value_cache;
    ov::Tensor scale;
    ov::Tensor sliding_window;
    ov::Tensor max_context_len;
    ov::Tensor past_lens;
    ov::Tensor subsequence_begins;
    ov::Tensor block_indices;
    ov::Tensor block_indices_begins;
    ov::Tensor alibi_slopes;
    ov::Tensor rotated_block_indices;
    ov::Tensor rotation_deltas;
    ov::Tensor rotation_trig_lut;
    std::string targetDevice;
};

TEST_P(ReferencePagedAttention, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_With_Hardcoded_Refs,
    ReferencePagedAttentionLayerTest,
    ::testing::Values(
        // Test case 1: No past tokens (all new); no rotation.
        PagedAttentionParams(
            // query
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}),
            // key
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // value
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache (empty)
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            // value_cache (empty)
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            // past_lens
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // subsequence_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 2}),
            // block_indices (empty)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            // block_indices_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 0}),
            // scale
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            // rotated_block_indices (none)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            // rotation_deltas (none)
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // expected output data
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.731f,
                                                       0.269f,
                                                       0.0f,
                                                       0.0f,
                                                       0.269f,
                                                       0.731f,
                                                       0.0f,
                                                       0.0f,
                                                       0.5f,
                                                       0.5f,
                                                       0.0f,
                                                       0.0f,
                                                       0.5f,
                                                       0.5f,
                                                       0.0f,
                                                       0.0f}),
            "Reference"),
        // Test case 2: One new token with past tokens from cache.
        PagedAttentionParams(
            // query
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // key
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            // value
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache
            reference_tests::Tensor({1, 2, 2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
            // value_cache
            reference_tests::Tensor({1, 2, 2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
            // past_lens
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{2}),
            // subsequence_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // block_indices
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // block_indices_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // scale
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            // rotated_block_indices (none)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            // rotation_deltas (none)
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // expected output data
            reference_tests::Tensor({1, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.576f, 0.424f, 0.0f, 0.0f, 0.333f, 0.0f, 0.333f, 0.333f}),
            "Reference"),
        // Test case 3: One new token with past tokens and RoPE applied (rotation).
        PagedAttentionParams(
            // query
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // key
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            // value
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache
            reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // value_cache
            reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // past_lens
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{1}),
            // subsequence_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // block_indices
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // block_indices_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // scale
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            // rotated_block_indices
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // rotation_deltas
            reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int>{1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // expected output data
            reference_tests::Tensor({1, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.689f, 0.311f, 0.0f, 0.0f, 0.067f, 0.311f, 0.0f, 0.0f}),
            "Reference"),
        // Test case 4: Multiple sequences.
        // (Seq0: 1 past token & 2 new tokens; Seq1: 2 past tokens & 1 new token)
        PagedAttentionParams(
            // query (3 tokens)
            reference_tests::Tensor({3, 8},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1, 0, 0, 0, 0, 1, 0, 0,  // Token0 (seq0)
                                        0, 1, 0, 0, 1, 0, 0, 0,  // Token1 (seq0)
                                        1, 1, 0, 0, 0, 0, 1, 0   // Token2 (seq1)
                                    }),
            // key
            reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // value
            reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache
            reference_tests::Tensor({3, 2, 1, 4},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1, 0, 0, 0, 0, 1, 0, 0,  // Block 0 for seq0
                                        1, 0, 0, 0, 0, 1, 0, 0,  // Block 1 for seq1
                                        0, 1, 0, 0, 1, 0, 0, 0   // Block 2 for seq1
                                    }),
            // value_cache
            reference_tests::Tensor({3, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0,
                                                                                       1, 0, 0, 0, 0, 1, 0, 0,
                                                                                       0, 1, 0, 0, 1, 0, 0, 0}),
            // past_lens
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{1, 2}),
            // subsequence_begins
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 2, 3}),
            // block_indices
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 1, 2}),
            // block_indices_begins
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 1, 3}),
            // scale
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            // rotated_block_indices (none)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            // rotation_deltas (none)
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // expected output data
            reference_tests::Tensor({3, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.844f, 0.155f, 0.0f, 0.0f, 0.155f, 0.844f, 0.0f,   0.0f,
                                                       0.424f, 0.576f, 0.0f, 0.0f, 0.576f, 0.424f, 0.0f,   0.0f,
                                                       0.576f, 1.0f,   0.0f, 0.0f, 0.424f, 0.0f,   0.576f, 0.0f}),
            "Reference"),
        // Test case 5: Past tokens with a nonzero sliding_window.
        PagedAttentionParams(
            // query
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // key
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // value
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            // key_cache
            reference_tests::Tensor({1, 2, 2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // value_cache
            reference_tests::Tensor({1, 2, 2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // past_lens
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{2}),
            // subsequence_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // block_indices
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            // block_indices_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // scale
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{1}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            // rotated_block_indices (none)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            // rotation_deltas (none)
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // expected output data
            reference_tests::Tensor({1, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.731f, 0.269f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}),
            "Reference"),
        // Test case 6: Nonzero alibi slopes.
        PagedAttentionParams(
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
            reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{1}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.1f, 0.2f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            reference_tests::Tensor({1, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.711f, 0.289f, 0.0f, 0.0f, 0.31f, 0.69f, 0.0f, 0.0f}),
            "Reference"),
        // Test case 7: Two past blocks (with block 1 rotated) and two new tokens.
        PagedAttentionParams(
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1,
                                        0,
                                        0,
                                        0,
                                        0,
                                        1,
                                        0,
                                        0,  // Token0
                                        0,
                                        1,
                                        0,
                                        0,
                                        1,
                                        0,
                                        0,
                                        0  // Token1
                                    }),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({2, 2, 2, 4},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1, 0, 0, 0, 0, 1, 0, 0,  // Block 0
                                        0, 1, 0, 0, 1, 0, 0, 0,  // Block 0 continued
                                        1, 1, 0, 0, 0, 1, 0, 0,  // Block 1 (rotated)
                                        0, 0, 1, 0, 1, 0, 0, 0   // Block 1 continued
                                    }),
            reference_tests::Tensor({2, 2, 2, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                                                                                       0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                                                                       0, 0, 0, 0, 1, 0, 1, 0, 0, 0}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{3}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 2}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{10, 11}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0, 2}),
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{11}),
            reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int>{2}),
            reference_tests::Tensor(
                {3, 4},
                ov::element::f32,
                std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.75f, 0.75f}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.643f,
                                                       0.267f,
                                                       0.0f,
                                                       0.0f,
                                                       0.5f,
                                                       0.5f,
                                                       0.0f,
                                                       0.0f,
                                                       0.424f,
                                                       0.576f,
                                                       0.0f,
                                                       0.0f,
                                                       0.576f,
                                                       0.424f,
                                                       0.0f,
                                                       0.0f}),
            "Reference"),
        // Test case 8: Multiple sequences with different new token counts.
        PagedAttentionParams(
            reference_tests::Tensor({3, 8},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1, 0, 0, 0, 0, 1, 0, 0,  // Token0 (seq0)
                                        0, 1, 0, 0, 1, 0, 0, 0,  // Token1 (seq1)
                                        1, 0, 0, 0, 0, 1, 0, 0   // Token2 (seq1)
                                    }),
            reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                                 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                                 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            reference_tests::Tensor({3, 2, 1, 4},
                                    ov::element::f32,
                                    std::vector<float>{
                                        1, 0, 0, 0, 0, 1, 0, 0,  // block 0
                                        0, 1, 0, 0, 1, 0, 0, 0,  // block 1
                                        1, 1, 0, 0, 0, 0, 1, 0   // block 2
                                    }),
            reference_tests::Tensor({3, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0,
                                                                                       0, 1, 0, 0, 1, 0, 0, 0,
                                                                                       1, 1, 0, 0, 0, 0, 1, 0}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{1, 2}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 1, 3}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 1, 2}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int>{0, 1, 3}),
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                                 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            "Reference"),
        // Test case 9: All ones (trivial uniform softmax), no past tokens.
        PagedAttentionParams(
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 2}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 0}),
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{1.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
            "Reference"),
        // Test case 10: No past tokens but with a scale factor of 2.0.
        PagedAttentionParams(
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            reference_tests::Tensor({0, 2, 1, 4}, ov::element::f32, std::vector<float>{}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 2}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 0}),
            reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{2.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{0}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int>{10}),
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int>{}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int>{0, 1}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            reference_tests::Tensor({2, 8},
                                    ov::element::f32,
                                    std::vector<float>{0.88f,
                                                       0.119f,
                                                       0.0f,
                                                       0.0f,
                                                       0.119f,
                                                       0.88f,
                                                       0.0f,
                                                       0.0f,
                                                       0.5f,
                                                       0.5f,
                                                       0.0f,
                                                       0.0f,
                                                       0.5f,
                                                       0.5f,
                                                       0.0f,
                                                       0.0f}),
            "Reference")),
    ReferencePagedAttentionLayerTest::getTestCaseName);
}  // namespace