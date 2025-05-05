// Copyright (C) 2018-2025 Intel Corporation
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
// 16: reference_tests::Tensor           -- free_block_indices (shape: [num_blocks])
// 17: reference_tests::Tensor           -- max_blocks (shape: [batch_size_in_sequences])
// 18: reference_tests::Tensor           -- output data to compare reference output with
// 19: std::string                       -- targetDevice
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
                                        reference_tests::Tensor,  // 16: free_block_indices
                                        reference_tests::Tensor,  // 17: max_blocks
                                        reference_tests::Tensor,  // 18: output data
                                        std::string>;             // 19: targetDevice
class ReferencePagedAttentionLayerTest : public testing::TestWithParam<PagedAttentionParams>,
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
        free_block_indices = std::get<16>(params).data;
        max_blocks = std::get<17>(params).data;
        targetDevice = std::get<19>(params);

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

        refOutData = {std::get<18>(params).data};
    }

    static std::string tensor2str(const reference_tests::Tensor& t) {
        std::ostringstream oss;
        oss << "[type: ";
        oss << t.type;
        oss << ", shape: ";
        oss << t.shape.to_string();
        oss << ", vals: ";
        oss << "(";

        size_t size = std::accumulate(t.shape.begin(), t.shape.end(), 1, std::multiplies<size_t>{});
        for (size_t i = 0; i < size; ++i) {
            if (t.type == ov::element::f32) {
                oss << static_cast<float*>(t.data.data())[i];
            } else if (t.type == ov::element::i32) {
                oss << static_cast<int32_t*>(t.data.data())[i];
            }

            if (i + 1 != size) {
                oss << ", ";
            }
        }
        oss << ")]";
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
        name << "free_block_indices=" << tensor2str(std::get<16>(obj.param)) << "_";
        name << "max_blocks=" << tensor2str(std::get<17>(obj.param)) << "_";
        name << "trgDev=" << std::get<19>(obj.param) << "_";

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
        const auto& free_block_indices = std::get<16>(params).data;
        const auto& max_blocks = std::get<17>(params).data;

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
                                                    rotation_trig_lut,
                                                    free_block_indices,
                                                    max_blocks};
        ov::ParameterVector inputParams;
        for (auto& input : funcInputs) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_shape()));
        }

        auto paged_attn = std::make_shared<ov::op::PagedAttentionExtension>(inputParams[0],
                                                                            inputParams[1],
                                                                            inputParams[2],
                                                                            inputParams[3],
                                                                            inputParams[4],
                                                                            inputParams[5],
                                                                            inputParams[6],
                                                                            inputParams[7],
                                                                            inputParams[8],
                                                                            inputParams[9],
                                                                            inputParams[10],
                                                                            inputParams[11],
                                                                            inputParams[12],
                                                                            inputParams[13],
                                                                            inputParams[14],
                                                                            inputParams[15],
                                                                            inputParams[16],
                                                                            inputParams[17]);

        // TODO add reference tests for second output.
        return std::make_shared<ov::Model>(ov::OutputVector{paged_attn->output(0)}, inputParams);
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
    ov::Tensor free_block_indices;
    ov::Tensor max_blocks;
    std::string targetDevice;
};

TEST_P(ReferencePagedAttentionLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_With_Hardcoded_Refs,
    ReferencePagedAttentionLayerTest,
    ::testing::Values(
        // ----- BASIC TESTS, ensure proper execution for different param sets -----
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
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
            // subsequence_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
            // block_indices (empty)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
            // block_indices_begins
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
            // scale
            reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
            // sliding_window
            reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
            // alibi_slopes
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
            // max_context_len
            reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
            // rotated_block_indices (none)
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
            // rotation_deltas (none)
            reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
            // rotation_trig_lut
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
            // free_block_indices
            reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
            // max_blocks
            reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{100}),
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
            "Reference_0"),

        /*
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
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    // subsequence_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // block_indices
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    // block_indices_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // scale
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    // sliding_window
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    // alibi_slopes
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    // max_context_len
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    // rotated_block_indices (none)
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    // rotation_deltas (none)
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // rotation_trig_lut
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    // expected output data
                    reference_tests::Tensor({1, 8},
                                            ov::element::f32,
                                            std::vector<float>{0.576f, 0.424f, 0.0f, 0.0f, 0.333f, 0.0f, 0.333f,
           0.333f}), "Reference_1"),
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
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    // subsequence_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // block_indices
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    // block_indices_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // scale
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    // sliding_window
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    // alibi_slopes
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    // max_context_len
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    // rotated_block_indices
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    // rotation_deltas
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{1}),
                    // rotation_trig_lut
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    // expected output data
                    reference_tests::Tensor({1, 8},
                                            ov::element::f32,
                                            std::vector<float>{0.689f, 0.311f, 0.0f, 0.0f, 0.067f, 0.311f, 0.0f, 0.0f}),
                    "Reference_2"),
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
                    reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
                    // value
                    reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}),
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
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{1, 2}),
                    // subsequence_begins
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
                    // block_indices
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                    // block_indices_begins
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 3}),
                    // scale
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    // sliding_window
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    // alibi_slopes
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    // max_context_len
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    // rotated_block_indices (none)
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    // rotation_deltas (none)
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // rotation_trig_lut
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    // expected output data
                    // reference_tests::Tensor({3, 8},
                    //                         ov::element::f32,
                    //                         std::vector<float>{0.844f, 0.155f, 0.0f, 0.0f, 0.155f, 0.844f, 0.0f,
           0.0f,
                    //                                            0.424f, 0.576f, 0.0f, 0.0f, 0.576f, 0.424f, 0.0f,
           0.0f,
                    //                                            0.576f, 1.0f,   0.0f, 0.0f, 0.424f, 0.0f,   0.576f,
           0.0f}), reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{0.844f, 0.155f, 0.0f, 0.0f,
           0.155f, 0.844f, 0.0f, 0.0f, 0.424f, 0.576f, 0.0f, 0.0f, 0.576f, 0.424f, 0.0f, 0.0f, 0.333f, 0.666f, 0.0f,
           0.0f, 0.666f, 0.333f, 0.0f, 0.0f}), "Reference_3"),
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
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    // subsequence_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // block_indices
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    // block_indices_begins
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // scale
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    // sliding_window
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                    // alibi_slopes
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    // max_context_len
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    // rotated_block_indices (none)
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    // rotation_deltas (none)
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    // rotation_trig_lut
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    // expected output data
                    reference_tests::Tensor({1, 8},
                                            ov::element::f32,
                                            std::vector<float>{0.731f, 0.269f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}),
                    "Reference_4"),
                // Test case 6: Nonzero alibi slopes.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
                    reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
                    reference_tests::Tensor({1, 8}, ov::element::f32, std::vector<float>{0, 1, 0, 0, 1, 0, 0, 0}),
                    reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
                    reference_tests::Tensor({1, 2, 1, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.1f, 0.2f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    reference_tests::Tensor({1, 8},
                                            ov::element::f32,
                                            std::vector<float>{0.711f, 0.289f, 0.0f, 0.0f, 0.31f, 0.69f, 0.0f, 0.0f}),
                    "Reference_5"),
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
                    reference_tests::Tensor({2, 2, 2, 4}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0,
           0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0}), reference_tests::Tensor({1},
           ov::element::i32, std::vector<int32_t>{3}), reference_tests::Tensor({2}, ov::element::i32,
           std::vector<int32_t>{0, 2}), reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{10, 11}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{11}),
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{2}),
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
                    "Reference_6"),
                // Test case 8: Multiple sequences with different new token counts.
                PagedAttentionParams(
                    reference_tests::Tensor({3, 8},
                                            ov::element::f32,
                                            std::vector<float>{
                                                1, 0, 0, 0, 0, 1, 0, 0,  // Token0 (seq0)
                                                0, 1, 0, 0, 1, 0, 0, 0,  // Token1 (seq1)
                                                1, 0, 0, 0, 0, 1, 0, 0   // Token2 (seq1)
                                            }),
                    reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}), reference_tests::Tensor({3, 8}, ov::element::f32,
           std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
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
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{1, 2}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 3}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 3}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    reference_tests::Tensor({3, 8}, ov::element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}), "Reference_7"),
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
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f}),
                    reference_tests::Tensor({2, 8},
                                            ov::element::f32,
                                            std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                    "Reference_8"),
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
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{2.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.0f, 0.0f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{10}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 1}),
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
                    "Reference_9"),
                // ----- ADVANCED TESTS, precomputed mathematically for rotation -----
                // Test case 10: Use default pointers (simulate nullptr for scale, sliding_window, alibi).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
                                     reference_tests::Tensor({0}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     "Reference_10"),

                // Test case 11: Multi–head with mixed cached and new tokens.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f, 1.f, 0.f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f, 1.f, 0.f}),
                    reference_tests::Tensor({1, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.f, 0.f}),
                    reference_tests::Tensor({1, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.f, 0.f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.f, 0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.55f, 0.45f, 1.f, 0.f}),
                    "Reference_11"),

                // Test case 12: Cached token falling in sliding window region is skipped for accumulation.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.5f,
           0.5f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.5f, 0.5f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     "Reference_12"),

                // Test case 13: Complex test with multiple blocks and rotation on one block only.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.69f, 0.45f}),
                    "Reference_13"),

                // Test case 14: Test with high scale factor.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f,
           0.6f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{2.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.69f,
           0.31f}), "Reference_14"),

                // Test case 15: Test with negative alibi slopes.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.9f,
           0.1f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.95f,
           0.05f}), "Reference_15"),

                // Test case 16: Test with non–trivial rotation trig LUT.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.5f,
           0.5f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.5f, 0.5f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, -0.6f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.804f,
           0.027f}), "Reference_16"),

                // Test case 17: Test with multiple rotated blocks.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{0, 0}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.744f, 0.414f}),
                    "Reference_17"),

                // Test case 18: Larger head size (head_size = 4).
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{1.f, 0.f, 1.f, 0.f}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{1.f, 0.f, 1.f, 0.f}),
                    reference_tests::Tensor({1, 1, 1, 4}, ov::element::f32, std::vector<float>{0.5f, 0.5f, 0.2f, 0.8f}),
                    reference_tests::Tensor({1, 1, 1, 4}, ov::element::f32, std::vector<float>{0.5f, 0.5f, 0.2f, 0.8f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.f, 0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 4}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.8f, 0.2f, 0.8f, 0.2f}),
                    "Reference_18"),

                // Test case 19: All tokens from new input (empty cache).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     "Reference_19"),

                // Test case 20: Cache and new tokens equally weighted.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.6f, 0.4f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f,
           0.7f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.63f,
           0.392f}), "Reference_20"),

                // Test case 21: Multiple queries in one batch with different settings.
                PagedAttentionParams(
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.5f, 0.5f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.5f, 0.5f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.5f, 0.5f}),
                    reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                    reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{1, 0}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{(1 + 1) + (0 + 1)}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.89f, 0.11f, 0.5f, 0.5f}),
                    "Reference_21"),

                // Test case 22: Sliding window excludes more cached tokens.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.3f, 0.7f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.3f, 0.7f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                    "Reference_22"),

                // Test case 23: Rotation with token offset selecting different trig index.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 1, 2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({1, 1, 2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.8f, 0.2f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1, 2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.75f, 0.35f}),
                    "Reference_23"),

                // Test case 24: Test with block_indices_begins not provided (simulate default).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.8f,
           0.2f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     "Reference_24"),

                // Test case 25: Test with scale pointer as nullptr (defaults to 1).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({0, 1, 1, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     "Reference_25"),

                // Test case 26: Test with sliding_window pointer as nullptr (defaults to 0).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.6f,
           0.4f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.6f, 0.4f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     "Reference_26"),

                // Test case 27: Test with alibi_slopes pointer as nullptr (defaults to 0).
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.7f,
           0.3f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({0}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     "Reference_27"),

                // Test case 28: Test with all rotation parameters non–null and triggered.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.6f,
           0.4f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.6f, 0.4f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, -0.6f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.987f,
           0.368f}), "Reference_28"),

                // Test case 29: Test with varying block sizes.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{1.f, 0.f}),
                    reference_tests::Tensor({1, 1, 2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.5f, 0.5f}),
                    reference_tests::Tensor({1, 1, 2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.5f, 0.5f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.78f, 0.22f}),
                    "Reference_29"),

                // Test case 30: Full complex scenario with two sequences, multi–heads, rotation, sliding window, and
           alibi
                // slopes.
                PagedAttentionParams(
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.f, 1.f, 0.8f, 0.2f, 0.2f, 0.8f}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f}),
                    reference_tests::Tensor({3, 2, 1, 2},
                                            ov::element::f32,
                                            std::vector<float>{0.5f, 0.5f, 0.8f, 0.2f, 0.7f, 0.3f}),
                    reference_tests::Tensor({3, 2, 1, 2},
                                            ov::element::f32,
                                            std::vector<float>{0.5f, 0.5f, 0.8f, 0.2f, 0.7f, 0.3f}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{2, 1}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.05f, 0.07f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{4}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, -0.6f}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{0.70f, 0.30f, 0.65f, 0.35f, 0.90f, 0.10f, 0.88f, 0.12f}),
                    "Reference_30"),
                // Test case 31: Complex multi‐head with non‐uniform cached tokens, non‐zero alibi slopes, rotation on
           block1
                // and sliding window exclusion.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.9f, 0.1f, 0.3f, 0.7f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.95f, 0.05f, 0.4f, 0.6f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.95f, 0.05f, 0.4f, 0.6f}),
                    reference_tests::Tensor({2, 2, 1, 2}, ov::element::f32, std::vector<float>{0.2f, 0.8f, 0.7f, 0.3f}),
                    reference_tests::Tensor({2, 2, 1, 2}, ov::element::f32, std::vector<float>{0.2f, 0.8f, 0.7f, 0.3f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.2f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.05f, -0.03f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),  // block1 is rotated
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.7f, 0.3f}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.755f, 0.181f, 0.400f,
           0.515f}), "Reference_31"),

                // Test case 32: Two sequences with multi‐head, differing past_lens and rotations on one sequence.
                PagedAttentionParams(
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.5f, 0.5f, 0.8f, 0.2f, 0.2f, 0.8f}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.5f, 0.5f, 1.f, 0.f, 1.f, 0.f}),
                    reference_tests::Tensor({2, 4},
                                            ov::element::f32,
                                            std::vector<float>{1.f, 0.f, 0.5f, 0.5f, 1.f, 0.f, 1.f, 0.f}),
                    reference_tests::Tensor({3, 2, 1, 2},
                                            ov::element::f32,
                                            std::vector<float>{0.4f, 0.6f, 0.9f, 0.1f, 0.7f, 0.3f}),
                    reference_tests::Tensor({3, 2, 1, 2},
                                            ov::element::f32,
                                            std::vector<float>{0.4f, 0.6f, 0.9f, 0.1f, 0.7f, 0.3f}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{2, 1}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                    reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.04f, -0.02f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{4}),
                    reference_tests::Tensor({0},
                                            ov::element::i32,
                                            std::vector<int32_t>{}),  // no rotation for seq0 cached tokens
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{0, 0}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),  // no rotation trig LUT
           used reference_tests::Tensor({2, 4}, ov::element::f32, std::vector<float>{0.85f, 0.15f, 0.67f, 0.33f, 0.9f,
           0.1f, 0.88f, 0.12f}), "Reference_32"),

                // Test case 33: Single query with high scale and negative alibi slope.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.55f,
           0.45f}), reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.65f, 0.35f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.65f,
           0.35f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.8f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.4f,
           0.8f}), reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}), reference_tests::Tensor({2},
           ov::element::i32, std::vector<int32_t>{0, 1}), reference_tests::Tensor({1}, ov::element::i32,
           std::vector<int32_t>{0}), reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{2.5f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{-0.07f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.511f,
           0.601f}), "Reference_33"),

                // Test case 34: Multi‐head with no cached tokens (empty cache).
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    reference_tests::Tensor({0, 2, 1, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({0, 2, 1, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 0}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.f, 0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    "Reference_34"),

                // Test case 35: Sliding window excludes all cached tokens.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     reference_tests::Tensor({3, 1, 1, 2},
                                                             ov::element::f32,
                                                             std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}),
                                     reference_tests::Tensor({3, 1, 1, 2},
                                                             ov::element::f32,
                                                             std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{3}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                                     reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
                                     reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 3}),
                                     reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                                     reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{4}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f}),
                                     "Reference_35"),

                // Test case 36: Rotation applied only for block1.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.5f, 0.5f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.5f, 0.5f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.2f, 0.8f, 0.9f, 0.1f}),
                    reference_tests::Tensor({2, 1, 1, 2}, ov::element::f32, std::vector<float>{0.2f, 0.8f, 0.9f, 0.1f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),  // block1 is rotated
                    reference_tests::Tensor({1, 1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.6f, 0.4f}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.394f, 0.580f}),
                    "Reference_36"),

                // Test case 37: Multi‐head with different alibi slopes per head and no rotation.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.2f, 0.8f, 0.9f, 0.1f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f, 0.6f, 0.4f}),
                    reference_tests::Tensor({1, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.7f, 0.3f}),
                    reference_tests::Tensor({1, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.7f, 0.3f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{0}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.02f, -0.05f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                    reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.348f, 0.652f, 0.653f,
           0.347f}), "Reference_37"),

                // Test case 38: Default block_indices_begins (simulate not provided) and no rotation triggered.
                PagedAttentionParams(reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.5f, 0.5f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f}),
                                     reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f,
           0.7f}), reference_tests::Tensor({1, 1, 1, 2}, ov::element::f32, std::vector<float>{0.3f, 0.7f}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{1}),
                                     reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{0}),
                                     reference_tests::Tensor({1},
                                                             ov::element::i32,
                                                             std::vector<int32_t>{0}),  // block_indices_begins not
           provided reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}), reference_tests::Tensor({},
           ov::element::f32, std::vector<float>{1.f}), reference_tests::Tensor({}, ov::element::i32,
           std::vector<int32_t>{0}), reference_tests::Tensor({1}, ov::element::f32, std::vector<float>{0.f}),
                                     reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{2}),
                                     reference_tests::Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 1}, ov::element::i32, std::vector<int32_t>{}),
                                     reference_tests::Tensor({0, 2}, ov::element::f32, std::vector<float>{}),
                                     reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.35f,
           0.65f}), "Reference_38"),

                // Test case 39: Complex with high scale, multi‐head, rotation and sliding window.
                PagedAttentionParams(
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.95f, 0.05f, 0.15f, 0.85f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f, 0.2f, 0.8f}),
                    reference_tests::Tensor({2, 2}, ov::element::f32, std::vector<float>{0.9f, 0.1f, 0.2f, 0.8f}),
                    reference_tests::Tensor({2, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.5f, 0.5f}),
                    reference_tests::Tensor({2, 2, 1, 2}, ov::element::f32, std::vector<float>{0.4f, 0.6f, 0.5f, 0.5f}),
                    reference_tests::Tensor({1}, ov::element::i32, std::vector<int32_t>{2}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 1}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 2}),
                    reference_tests::Tensor({}, ov::element::f32, std::vector<float>{3.f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
                    reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.1f, -0.1f}),
                    reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{3}),
                    reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{1, 2}),  // blocks 1 and 2
           rotated reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{1, 1}),
                    reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
                    reference_tests::Tensor({1, 4}, ov::element::f32, std::vector<float>{0.811f, 0.159f, 0.16f,
           0.728f}), "Reference_39"),
        */
        // Test case 40: Full complex scenario with two sequences, multi‐heads, rotations on both sequences, sliding
        // window, and non‐zero alibi slopes.
        PagedAttentionParams(
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{0.6f, 0.4f, 0.3f, 0.7f, 0.2f, 0.8f, 0.9f, 0.1f}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{0.5f, 0.5f, 0.4f, 0.6f, 0.8f, 0.2f, 0.7f, 0.3f}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{0.5f, 0.5f, 0.4f, 0.6f, 0.8f, 0.2f, 0.7f, 0.3f}),
            reference_tests::Tensor({3, 2, 1, 2},
                                    ov::element::f32,
                                    std::vector<float>{
                                        0.4f,
                                        0.6f,
                                        0.7f,
                                        0.3f,  // seq0, block0
                                        0.9f,
                                        0.1f,
                                        0.3f,
                                        0.7f,  // seq0, block1 (rotated)
                                        0.5f,
                                        0.5f,
                                        0.6f,
                                        0.4f  // seq1, block0 (rotated)
                                    }),
            reference_tests::Tensor(
                {3, 2, 1, 2},
                ov::element::f32,
                std::vector<float>{0.4f, 0.6f, 0.7f, 0.3f, 0.9f, 0.1f, 0.3f, 0.7f, 0.5f, 0.5f, 0.6f, 0.4f}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{2, 1}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 1, 2}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 2, 3}),
            reference_tests::Tensor({}, ov::element::f32, std::vector<float>{1.f}),
            reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{1}),
            reference_tests::Tensor({2}, ov::element::f32, std::vector<float>{0.05f, -0.05f}),
            reference_tests::Tensor({}, ov::element::i32, std::vector<int32_t>{4}),
            reference_tests::Tensor({2},
                                    ov::element::i32,
                                    std::vector<int32_t>{1, 2}),  // both block1 and block2 rotated
            reference_tests::Tensor({2, 1}, ov::element::i32, std::vector<int32_t>{1, 1}),
            reference_tests::Tensor({1, 2}, ov::element::f32, std::vector<float>{0.8f, 0.2f}),
            reference_tests::Tensor({2}, ov::element::i32, std::vector<int32_t>{-1, -1}),
            reference_tests::Tensor({3}, ov::element::i32, std::vector<int32_t>{10, 10}),
            reference_tests::Tensor({2, 4},
                                    ov::element::f32,
                                    std::vector<float>{0.65f, 0.35f, 0.60f, 0.40f, 0.85f, 0.15f, 0.88f, 0.12f}),
            "Reference_40")),
    ReferencePagedAttentionLayerTest::getTestCaseName);
}  // namespace
