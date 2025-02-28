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

// 0: ov::Tensor            -- query
// 1: ov::Tensor            -- key
// 2: ov::Tensor            -- value
// 3: ov::Tensor            -- key_cache
// 4: ov::Tensor            -- value_cache
// 5: ov::Tensor            -- past_lens (shape: [batch_size_in_sequences])
// 6: ov::Tensor            -- subsequence_begins (shape: [batch_size_in_sequences + 1])
// 7: ov::Tensor            -- block_indices (shape: [num_blocks])
// 8: ov::Tensor            -- block_indices_begins (shape: [batch_size_in_sequences + 1])
// 9: ov::Tensor            -- scale
// 10: ov::Tensor           -- sliding_window
// 11: ov::Tensor           -- alibi_slopes (shape: [num_kv_heads])
// 12: ov::Tensor           -- max_context_len
// 13: ov::Tensor           -- rotated_block_indices (shape: [num_rotated_blocks])
// 14: ov::Tensor           -- rotation_deltas (shape: as specified, e.g. [num_rotated_blocks, 1])
// 15: ov::Tensor           -- rotation_trig_lut (shape: [lut_rows, head_size])
// 16: ov::Tensor           -- output data to compare reference output with
// 17: std::string          -- targetDevice
using PagedAttentionParams = std::tuple<ov::Tensor,    // 0: query
                                        ov::Tensor,    // 1: key
                                        ov::Tensor,    // 2: value
                                        ov::Tensor,    // 3: key_cache
                                        ov::Tensor,    // 4: value_cache
                                        ov::Tensor,    // 5: past_lens
                                        ov::Tensor,    // 6: subsequence_begins
                                        ov::Tensor,    // 7: block_indices
                                        ov::Tensor,    // 8: block_indices_begins
                                        ov::Tensor,    // 9: scale
                                        ov::Tensor,    // 10: sliding_window
                                        ov::Tensor,    // 11: alibi_slopes
                                        ov::Tensor,    // 12: max_context_len
                                        ov::Tensor,    // 13: rotated_block_indices
                                        ov::Tensor,    // 14: rotation_deltas
                                        ov::Tensor,    // 15: rotation_trig_lut
                                        ov::Tensor,    // 16: output data
                                        std::string>;  // 17: targetDevice
class ReferencePagedAttention : public testing::TestWithParam<PagedAttentionParams>,
                                public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        query = std::get<0>(params);
        key = std::get<1>(params);
        value = std::get<2>(params);
        key_cache = std::get<3>(params);
        value_cache = std::get<4>(params);
        past_lens = std::get<5>(params);
        subsequence_begins = std::get<6>(params);
        block_indices = std::get<7>(params);
        block_indices_begins = std::get<8>(params);
        scale = std::get<9>(params);
        sliding_window = std::get<10>(params);
        alibi_slopes = std::get<11>(params);
        max_context_len = std::get<12>(params);
        rotated_block_indices = std::get<13>(params);
        rotation_deltas = std::get<14>(params);
        rotation_trig_lut = std::get<15>(params);
        targetDevice = std::get<17>(params);

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

    static std::string tensor2str(const ov::Tensor& t) {
        std::ostringstream oss;
        oss << "[type: ";
        oss << t.get_element_type();
        oss << ", shape: ";
        oss << t.get_shape().to_string();
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
        const auto& query = std::get<0>(params);
        const auto& key = std::get<1>(params);
        const auto& value = std::get<2>(params);
        const auto& key_cache = std::get<3>(params);
        const auto& value_cache = std::get<4>(params);
        const auto& scale = std::get<9>(params);
        const auto& sliding_window = std::get<10>(params);
        const auto& max_context_len = std::get<12>(params);
        const auto& past_lens = std::get<5>(params);
        const auto& subsequence_begins = std::get<6>(params);
        const auto& block_indices = std::get<7>(params);
        const auto& block_indices_begins = std::get<8>(params);
        const auto& alibi_slopes = std::get<11>(params);
        const auto& rotated_block_indices = std::get<13>(params);
        const auto& rotation_deltas = std::get<14>(params);
        const auto& rotation_trig_lut = std::get<15>(params);

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
            // query: two tokens, two heads (head_size=4)
            // Token0: head0 = [1,0,0,0], head1 = [0,1,0,0]
            // Token1: head0 = [1,1,0,0], head1 = [0,0,1,0]
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}),
            // key (new tokens):
            // token0: head0 = [1,0,0,0], head1 = [0,1,0,0]
            // token1: head0 = [0,1,0,0], head1 = [1,0,0,0]
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // value (new tokens) – same as key in this case.
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache is empty because past_lens = 0.
            ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
            ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
            // past_lens: 0 past tokens.
            ov::Tensor(ov::element::i32, {1}, {0}),
            // subsequence_begins: new tokens [0,2)
            ov::Tensor(ov::element::i32, {2}, {0, 2}),
            // block_indices (empty).
            ov::Tensor(ov::element::i32, {0}, {}),
            // block_indices_begins.
            ov::Tensor(ov::element::i32, {2}, {0, 0}),
            // scale = 1.0.
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            // sliding_window = 0.
            ov::Tensor(ov::element::i32, {1}, {0}),
            // alibi_slopes (2 kv heads).
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            // max_context_len = 10.
            ov::Tensor(ov::element::i32, {1}, {10}),
            // rotated_block_indices (none).
            ov::Tensor(ov::element::i32, {0}, {}),
            // rotation_deltas (none).
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            // dummy trig LUT (unused here).
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            // Expected output computed per head:
            // Token0: head0 ≈ [0.731, 0.269, 0, 0], head1 ≈ [0.269, 0.731, 0, 0]
            // Token1: head0 = [0.5, 0.5, 0, 0], head1 = [0.5, 0.5, 0, 0]
            ov::Tensor(ov::element::f32,
                       {2, 8},
                       {0.731, 0.269, 0.0, 0.0, 0.269, 0.731, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0}),
            "Reference"),
        // Test case 2: One new token with past tokens from cache.
        PagedAttentionParams(
            // query token: head0 = [1,0,0,0], head1 = [0,1,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
            // new key: head0 = [0,1,0,0], head1 = [1,0,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
            // new value: same as key.
            ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache: one block, 2 tokens per block.
            // Head0: token0 = [1,1,0,0], token1 = [0,1,0,0]
            // Head1: token0 = [0,0,1,0], token1 = [0,0,0,1]
            ov::Tensor(ov::element::f32, {1, 2, 2, 4}, {1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
            ov::Tensor(ov::element::f32, {1, 2, 2, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
            ov::Tensor(ov::element::i32, {1}, {2}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            ov::Tensor(ov::element::i32, {0}, {}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            ov::Tensor(ov::element::f32, {1, 8}, {0.576, 0.424, 0.0, 0.0, 0.333, 0.0, 0.333, 0.333}),
            "Reference"),
        // Test case 3: One new token with past tokens and RoPE applied (rotation).
        PagedAttentionParams(
            // query: head0 = [1,0,0,0], head1 = [0,1,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
            // new key: head0 = [0,1,0,0], head1 = [1,0,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
            // new value: same as key.
            ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache: one block for two heads.
            // Head0: [1,0,0,0]
            // Head1: [0,1,0,0]
            ov::Tensor(ov::element::f32, {1, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0}),
            ov::Tensor(ov::element::f32, {1, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0}),
            ov::Tensor(ov::element::i32, {1}, {1}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            // Rotate block 0.
            ov::Tensor(ov::element::i32, {1}, {0}),
            // rotation_deltas: trig index = 1.
            ov::Tensor(ov::element::i32, {1, 1}, {1}),
            // rotation_trig_lut: two rows (row0 unused, row1 used)
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            ov::Tensor(ov::element::f32, {1, 8}, {0.689, 0.311, 0.0, 0.0, 0.067, 0.311, 0.0, 0.0}),
            "Reference"),
        // Test case 4: Multiple sequences.
        // (Seq0 with 1 past token & 2 new tokens; Seq1 with 2 past tokens & 1 new token)
        PagedAttentionParams(
            // Three tokens total.
            ov::Tensor(ov::element::f32, {3, 8}, {1, 0, 0, 0, 0, 1, 0, 0,    // Token0 (seq0)
                                                  0, 1, 0, 0, 1, 0, 0, 0,    // Token1 (seq0)
                                                  1, 1, 0, 0, 0, 0, 1, 0}),  // Token2 (seq1)
            ov::Tensor(ov::element::f32, {3, 8}, {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::f32, {3, 8}, {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            // key_cache: 3 blocks (each one token per block)
            ov::Tensor(ov::element::f32, {3, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0,    // Block 0 for seq0
                                                        1, 0, 0, 0, 0, 1, 0, 0,    // Block 1 for seq1
                                                        0, 1, 0, 0, 1, 0, 0, 0}),  // Block 2 for seq1
            ov::Tensor(ov::element::f32, {3, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                                                        0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::i32, {2}, {1, 2}),
            // subsequence_begins: seq0 tokens [0,2), seq1 tokens [2,3)
            ov::Tensor(ov::element::i32, {3}, {0, 2, 3}),
            // block_indices: seq0 uses block0; seq1 uses blocks 1 & 2.
            ov::Tensor(ov::element::i32, {3}, {0, 1, 2}),
            // block_indices_begins: seq0: [0,1), seq1: [1,3)
            ov::Tensor(ov::element::i32, {3}, {0, 1, 3}),
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            ov::Tensor(ov::element::i32, {0}, {}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            ov::Tensor(ov::element::f32, {3, 8}, {0.844, 0.155, 0.0, 0.0, 0.155, 0.844, 0.0,   0.0,
                                                  0.424, 0.576, 0.0, 0.0, 0.576, 0.424, 0.0,   0.0,
                                                  0.576, 1.0,   0.0, 0.0, 0.424, 0.0,   0.576, 0.0}),
            "Reference"),
        // Test case 5: Past tokens with a nonzero sliding_window.
        PagedAttentionParams(
            // query: [1,0,0,0, 0,1,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
            // key: [1,0,0,0, 0,1,0,0]
            ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
            // value: same as key.
            ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
            // key_cache: one block, 2 tokens.
            // Head0: [1,0,0,0] then [0,1,0,0]
            // Head1: [0,1,0,0] then [1,0,0,0]
            ov::Tensor(ov::element::f32, {1, 2, 2, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::f32, {1, 2, 2, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::i32, {1}, {2}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            ov::Tensor(ov::element::i32, {1}, {1}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            ov::Tensor(ov::element::i32, {0}, {}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            ov::Tensor(ov::element::f32, {1, 8}, {0.731, 0.269, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
            "Reference"),
        // Test case 6: Nonzero alibi slopes.
        PagedAttentionParams(ov::Tensor(ov::element::f32, {1, 8}, {1, 0, 0, 0, 0, 1, 0, 0}),
                             ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
                             ov::Tensor(ov::element::f32, {1, 8}, {0, 1, 0, 0, 1, 0, 0, 0}),
                             ov::Tensor(ov::element::f32, {1, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0}),
                             ov::Tensor(ov::element::f32, {1, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0}),
                             ov::Tensor(ov::element::i32, {1}, {1}),
                             ov::Tensor(ov::element::i32, {2}, {0, 1}),
                             ov::Tensor(ov::element::i32, {1}, {0}),
                             ov::Tensor(ov::element::i32, {2}, {0, 1}),
                             ov::Tensor(ov::element::f32, {1}, {1.0}),
                             ov::Tensor(ov::element::i32, {1}, {0}),
                             ov::Tensor(ov::element::f32, {2}, {0.1, 0.2}),
                             ov::Tensor(ov::element::i32, {1}, {10}),
                             ov::Tensor(ov::element::i32, {0}, {}),
                             ov::Tensor(ov::element::i32, {2}, {0, 1}),
                             ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
                             ov::Tensor(ov::element::f32, {1, 8}, {0.711, 0.289, 0.0, 0.0, 0.31, 0.69, 0.0, 0.0}),
                             "Reference"),
        // Test case 7: Two past blocks (with block 1 rotated) and two new tokens.
        PagedAttentionParams(
            ov::Tensor(ov::element::f32,
                       {2, 8},
                       {1,
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
                        0}),  // Token1
            ov::Tensor(ov::element::f32, {2, 8}, {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            ov::Tensor(ov::element::f32, {2, 8}, {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
            // key_cache: 2 blocks, each with 2 tokens.
            // Block 0: block_indices[0] = 10 (not rotated)
            // Head0: [1,0,0,0], [0,1,0,0]
            // Head1: [0,1,0,0], [1,0,0,0]
            // Block 1: block_indices[1] = 11 (rotated)
            // Head0: [1,1,0,0], [0,1,0,0]
            // Head1: [0,0,1,0], [1,0,0,0]
            ov::Tensor(ov::element::f32, {2, 2, 2, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                                                        1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::f32, {2, 2, 2, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                                                        1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::i32, {1}, {3}),
            ov::Tensor(ov::element::i32, {2}, {0, 2}),
            ov::Tensor(ov::element::i32, {2}, {10, 11}),
            ov::Tensor(ov::element::i32, {1}, {0, 2}),
            ov::Tensor(ov::element::f32, {1}, {1.0}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            // Rotate only block with id 11.
            ov::Tensor(ov::element::i32, {1}, {11}),
            ov::Tensor(ov::element::i32, {1, 1}, {2}),
            ov::Tensor(ov::element::f32, {3, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75}),
            ov::Tensor(ov::element::f32,
                       {2, 8},
                       {0.643, 0.267, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.424, 0.576, 0.0, 0.0, 0.576, 0.424, 0.0, 0.0}),
            "Reference"),
        // Test case 8: Multiple sequences with different new token counts.
        PagedAttentionParams(ov::Tensor(ov::element::f32, {3, 8}, {1, 0, 0, 0, 0, 1, 0, 0,    // Token0 (seq0)
                                                                   0, 1, 0, 0, 1, 0, 0, 0,    // Token1 (seq1)
                                                                   1, 0, 0, 0, 0, 1, 0, 0}),  // Token2 (seq1)
                             ov::Tensor(ov::element::f32, {3, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                   1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
                             ov::Tensor(ov::element::f32, {3, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                   1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
                             // key_cache: 3 blocks, 2 heads, 1 token each.
                             ov::Tensor(ov::element::f32, {3, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0,    // block 0
                                                                         0, 1, 0, 0, 1, 0, 0, 0,    // block 1
                                                                         1, 1, 0, 0, 0, 0, 1, 0}),  // block 2
                             ov::Tensor(ov::element::f32, {3, 2, 1, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                         1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}),
                             ov::Tensor(ov::element::i32, {2}, {1, 2}),
                             ov::Tensor(ov::element::i32, {3}, {0, 1, 3}),
                             ov::Tensor(ov::element::i32, {3}, {0, 1, 2}),
                             ov::Tensor(ov::element::i32, {3}, {0, 1, 3}),
                             ov::Tensor(ov::element::f32, {1}, {1.0}),
                             ov::Tensor(ov::element::i32, {1}, {0}),
                             ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
                             ov::Tensor(ov::element::i32, {1}, {10}),
                             ov::Tensor(ov::element::i32, {0}, {}),
                             ov::Tensor(ov::element::i32, {2}, {0, 1}),
                             ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
                             ov::Tensor(ov::element::f32, {3, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                                   1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
                             "Reference"),
        // Test case 9: All ones (trivial uniform softmax), no past tokens.
        PagedAttentionParams(ov::Tensor(ov::element::f32, {2, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                             ov::Tensor(ov::element::f32, {2, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                             ov::Tensor(ov::element::f32, {2, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                             ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
                             ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
                             ov::Tensor(ov::element::i32, {1}, {0}),
                             ov::Tensor(ov::element::i32, {2}, {0, 2}),
                             ov::Tensor(ov::element::i32, {0}, {}),
                             ov::Tensor(ov::element::i32, {2}, {0, 0}),
                             ov::Tensor(ov::element::f32, {1}, {1.0}),
                             ov::Tensor(ov::element::i32, {1}, {0}),
                             ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
                             ov::Tensor(ov::element::i32, {1}, {10}),
                             ov::Tensor(ov::element::i32, {0}, {}),
                             ov::Tensor(ov::element::i32, {2}, {0, 1}),
                             ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
                             ov::Tensor(ov::element::f32, {2, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                             "Reference"),
        // Test case 10: No past tokens but with a scale factor of 2.0.
        PagedAttentionParams(
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}),
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::f32, {2, 8}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
            ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
            ov::Tensor(ov::element::f32, {0, 2, 1, 4}, {}),
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::i32, {2}, {0, 2}),
            ov::Tensor(ov::element::i32, {0}, {}),
            ov::Tensor(ov::element::i32, {2}, {0, 0}),
            ov::Tensor(ov::element::f32, {1}, {2.0}),  // scale = 2.0
            ov::Tensor(ov::element::i32, {1}, {0}),
            ov::Tensor(ov::element::f32, {2}, {0.0, 0.0}),
            ov::Tensor(ov::element::i32, {1}, {10}),
            ov::Tensor(ov::element::i32, {0}, {}),
            ov::Tensor(ov::element::i32, {2}, {0, 1}),
            ov::Tensor(ov::element::f32, {2, 4}, {1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5}),
            ov::Tensor(ov::element::f32,
                       {2, 8},
                       {0.88, 0.119, 0.0, 0.0, 0.119, 0.88, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0}),
            "Reference")),
    ReferencePagedAttentionLayerTest::getTestCaseName);
}  // namespace