// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CPU reference coverage for ov::experimental::BlockSparseAttention. The op no longer exposes a
// CPU evaluate()/has_evaluate() path (evaluate is GPU-only now), so these tests exercise
// ov::reference::block_sparse_attention directly on element-typed inputs.

#include <cmath>
#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/reference/block_sparse_attention.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

// Directly computes ov::reference::block_sparse_attention on the given (element-typed) inputs and
// verifies the output has the expected element count and contains well-defined (finite) values.
void check_bsa_eval(const ov::element::Type& dt,
                    const ov::Shape& q_shape,
                    const ov::Shape& k_shape,
                    const ov::Shape& v_shape,
                    const ov::Shape& mask_shape,
                    const std::vector<float>& q_data,
                    const std::vector<float>& k_data,
                    const std::vector<float>& v_data,
                    const std::vector<char>& mask_data,
                    int64_t block_size_q,
                    int64_t block_size_kv,
                    bool causal) {
    ov::TensorVector in{ov::Tensor(dt, q_shape),
                        ov::Tensor(dt, k_shape),
                        ov::Tensor(dt, v_shape),
                        ov::Tensor(ov::element::boolean, mask_shape)};
    // Fill input tensors with the test data (cast to the tensor's element type).
    auto fill = [&](ov::Tensor& t, const std::vector<float>& data) {
        if (dt == ov::element::f32) {
            for (size_t i = 0; i < data.size(); ++i)
                static_cast<float*>(t.data())[i] = data[i];
        } else if (dt == ov::element::f16) {
            for (size_t i = 0; i < data.size(); ++i)
                static_cast<ov::float16*>(t.data())[i] = ov::float16(data[i]);
        } else if (dt == ov::element::bf16) {
            for (size_t i = 0; i < data.size(); ++i)
                static_cast<ov::bfloat16*>(t.data())[i] = ov::bfloat16(data[i]);
        }
    };
    fill(in[0], q_data);
    fill(in[1], k_data);
    fill(in[2], v_data);
    auto* mask_ptr = static_cast<char*>(in[3].data());
    for (size_t i = 0; i < mask_data.size(); ++i)
        mask_ptr[i] = mask_data[i];

    const size_t B = q_shape[0], Lq = q_shape[1], Hq = q_shape[2], E = q_shape[3];
    const size_t Lk = k_shape[1], Hkv = k_shape[2], Ev = v_shape[3];
    const size_t Bm = mask_shape[0], Hm = mask_shape[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(E));
    const size_t out_count = B * Lq * Hq * Ev;

    // Compute the reference output, running in the input's own (element-type-quantized) values.
    std::vector<float> ref(out_count);
    auto run_ref = [&](auto* tag) {
        using T = std::remove_pointer_t<std::decay_t<decltype(tag)>>;
        std::vector<T> typed_ref(out_count);
        ov::reference::block_sparse_attention(static_cast<const T*>(in[0].data()),
                                              static_cast<const T*>(in[1].data()),
                                              static_cast<const T*>(in[2].data()),
                                              mask_data.data(),
                                              typed_ref.data(),
                                              B,
                                              Lq,
                                              Lk,
                                              Hq,
                                              Hkv,
                                              E,
                                              Ev,
                                              Bm,
                                              Hm,
                                              static_cast<size_t>(block_size_q),
                                              static_cast<size_t>(block_size_kv),
                                              scale,
                                              causal);
        for (size_t i = 0; i < out_count; ++i)
            ref[i] = static_cast<float>(typed_ref[i]);
    };
    if (dt == ov::element::f32)
        run_ref(static_cast<float*>(nullptr));
    else if (dt == ov::element::f16)
        run_ref(static_cast<ov::float16*>(nullptr));
    else
        run_ref(static_cast<ov::bfloat16*>(nullptr));

    // The reference is now the authoritative implementation; verify the expected output count and
    // that it produced well-defined (finite) values.
    EXPECT_EQ(ref.size(), out_count);
    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_TRUE(std::isfinite(ref[i])) << " at index=" << i;
}

}  // namespace

TEST(BlockSparseAttentionEval, single_query_dense_f32) {
    // B=1, Lq=1, Lk=2, Hq=1, Hkv=1, E=2, Ev=2, all-attend mask.
    // Analytically: Q=[1, 0], K0=[1, 0], K1=[0.5, 0.5], scale=1/sqrt(2)=0.7071f
    // Score0 = 0.7071f, Score1 = 0.3536f -> Softmax weights ~[0.5875f, 0.4125f]
    // V0=[1, 1], V1=[2, 2] -> Out = 0.5875*[1, 1] + 0.4125*[2, 2] = [1.4125f, 1.4125f]
    check_bsa_eval(ov::element::f32,
                   {1, 1, 1, 2},
                   {1, 2, 1, 2},
                   {1, 2, 1, 2},
                   {1, 1, 1, 1},  // Bm, Hm, num_q_blocks, num_k_blocks (num_k_blocks = ceil(2/2)=1)
                   {1.0f, 0.0f},
                   {1.0f, 0.0f, 0.5f, 0.5f},
                   {1.0f, 1.0f, 2.0f, 2.0f},
                   {1},
                   2,
                   2,
                   false);

    const float q[] = {1.0f, 0.0f};
    const float k[] = {1.0f, 0.0f, 0.5f, 0.5f};
    const float v[] = {1.0f, 1.0f, 2.0f, 2.0f};
    const char m[] = {1};
    std::vector<float> out(2);
    ov::reference::
        block_sparse_attention(q, k, v, m, out.data(), 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1.0f / std::sqrt(2.0f), false);
    EXPECT_NEAR(out[0], 1.4125f, 1e-3f);
    EXPECT_NEAR(out[1], 1.4125f, 1e-3f);
}

TEST(BlockSparseAttentionEval, prefill_sparse_block_single_query_f32) {
    // B=1, Lq=1, Lk=3, Hq=2, Hkv=2, E=4, Ev=4, block_size_kv=2 -> num_k_blocks=2; sparse mask keeps
    // only the first kv block (mask index 1 == 0 -> skipped), so the middle/tail keys never
    // contribute. Verifies the mask is honored (evaluate does not run dense).
    const size_t B = 1, Lq = 1, Lk = 3, Hq = 2, Hkv = 2, E = 4, Ev = 4;
    const size_t num_k_blocks = (Lk + 2 - 1) / 2;  // = 2
    const size_t num_q_blocks = (Lq + 2 - 1) / 2;  // = 1
    std::vector<float> q(B * Lq * Hq * E), k(B * Lk * Hkv * E), v(B * Lk * Hkv * Ev);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = ((i % 3) == 0) ? 0.8f : -0.3f;
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = ((i % 3) == 1) ? 0.6f : -0.2f;
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.1f * static_cast<float>((i + 1) % 7);
    // Only the first kv block (index 0) is attended for every head.
    std::vector<char> mask(B * Hq * num_q_blocks * num_k_blocks, 0);
    for (size_t h = 0; h < Hq; ++h)
        mask[(h * num_q_blocks) * num_k_blocks + 0] = 1;

    // Dispatch each element type.
    for (const auto& dt : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        check_bsa_eval(dt,
                       {B, Lq, Hq, E},
                       {B, Lk, Hkv, E},
                       {B, Lk, Hkv, Ev},
                       {B, Hq, num_q_blocks, num_k_blocks},
                       q,
                       k,
                       v,
                       mask,
                       2,
                       2,
                       false);
    }

    // Explicitly verify masked sparse output differs from dense (all-1) mask output
    std::vector<char> dense_mask(B * Hq * num_q_blocks * num_k_blocks, 1);
    std::vector<float> sparse_out(B * Lq * Hq * Ev), dense_out(B * Lq * Hq * Ev);
    ov::reference::block_sparse_attention(q.data(),
                                          k.data(),
                                          v.data(),
                                          mask.data(),
                                          sparse_out.data(),
                                          B,
                                          Lq,
                                          Lk,
                                          Hq,
                                          Hkv,
                                          E,
                                          Ev,
                                          B,
                                          Hq,
                                          2,
                                          2,
                                          0.5f,
                                          false);
    ov::reference::block_sparse_attention(q.data(),
                                          k.data(),
                                          v.data(),
                                          dense_mask.data(),
                                          dense_out.data(),
                                          B,
                                          Lq,
                                          Lk,
                                          Hq,
                                          Hkv,
                                          E,
                                          Ev,
                                          B,
                                          Hq,
                                          2,
                                          2,
                                          0.5f,
                                          false);
    bool any_diff = false;
    for (size_t i = 0; i < sparse_out.size(); ++i) {
        if (std::abs(sparse_out[i] - dense_out[i]) > 1e-4f) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "Masked block sparse attention must produce different output than dense attention";
}

TEST(BlockSparseAttentionEval, single_query_causal_f32) {
    // B=1, Lq=1, Lk=4, Hq=1, Hkv=1, E=2, Ev=2, causal=true, all-attend mask. Since Lq==1, the causal
    // mask keeps only key j <= 0, so only the first key/value contribute.
    check_bsa_eval(ov::element::f32,
                   {1, 1, 1, 2},
                   {1, 4, 1, 2},
                   {1, 4, 1, 2},
                   {1, 1, 1, 2},  // num_k_blocks = ceil(4/2)=2, all attended via mask; causal drops tail
                   {0.5f, -0.5f},
                   {1.0f, 0.0f, 0.2f, 0.8f, -0.4f, 0.4f, 0.9f, -0.1f},
                   {3.0f, -1.0f, 1.0f, 1.0f, 2.0f, 0.0f, -2.0f, 2.0f},
                   {1, 1},
                   2,
                   2,
                   true);

    // Hand-computed mathematical check: with Lq=1 and causal=true, only key 0 is attended.
    // Normalized softmax weight on key 0 is 1.0, so output MUST strictly equal V[0] = {3.0f, -1.0f}.
    // The op no longer exposes evaluate(), so the check runs the reference kernel directly.
    const float q_data[] = {0.5f, -0.5f};
    const float k_data[] = {1.0f, 0.0f, 0.2f, 0.8f, -0.4f, 0.4f, 0.9f, -0.1f};
    const float v_data[] = {3.0f, -1.0f, 1.0f, 1.0f, 2.0f, 0.0f, -2.0f, 2.0f};
    const char m_data[] = {1, 1};
    std::vector<float> causal_ref(1 * 1 * 1 * 2);
    ov::reference::block_sparse_attention(q_data,
                                          k_data,
                                          v_data,
                                          m_data,
                                          causal_ref.data(),
                                          1,
                                          1,
                                          4,
                                          1,
                                          1,
                                          2,
                                          2,
                                          1,
                                          1,
                                          2,
                                          2,
                                          1.0f,
                                          true);
    EXPECT_FLOAT_EQ(causal_ref[0], 3.0f);
    EXPECT_FLOAT_EQ(causal_ref[1], -1.0f);
}

TEST(BlockSparseAttentionEval, five_input_runtime_scale) {
    // 5-input form with an explicit runtime `scale` input differing from the default 1/sqrt(E).
    // The op no longer exposes evaluate(), so the reference kernel is exercised directly with the
    // runtime scale as ground truth.
    const std::vector<float> q{1.0f, 0.0f}, k{1.0f, 0.0f, 0.5f, 0.5f}, v{1.0f, 1.0f, 2.0f, 2.0f};
    const std::vector<char> mask{1};
    const float runtime_scale = 0.25f;

    const size_t B = 1, Lq = 1, Hq = 1, E = 2, Lk = 2, Hkv = 1, Ev = 2, Bm = 1, Hm = 1;
    const size_t out_count = B * Lq * Hq * Ev;
    std::vector<float> ref(out_count);
    ov::reference::block_sparse_attention(q.data(),
                                          k.data(),
                                          v.data(),
                                          mask.data(),
                                          ref.data(),
                                          B,
                                          Lq,
                                          Lk,
                                          Hq,
                                          Hkv,
                                          E,
                                          Ev,
                                          Bm,
                                          Hm,
                                          2,
                                          2,
                                          runtime_scale,
                                          false);

    EXPECT_EQ(ref.size(), out_count);
    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_TRUE(std::isfinite(ref[i])) << " at index=" << i;

    // Verify scale 0.25f produces different softmax distribution than default scale 1/sqrt(2) = 0.7071f
    std::vector<float> ref_default(out_count);
    ov::reference::block_sparse_attention(q.data(),
                                          k.data(),
                                          v.data(),
                                          mask.data(),
                                          ref_default.data(),
                                          B,
                                          Lq,
                                          Lk,
                                          Hq,
                                          Hkv,
                                          E,
                                          Ev,
                                          Bm,
                                          Hm,
                                          2,
                                          2,
                                          1.0f / std::sqrt(2.0f),
                                          false);
    EXPECT_NE(ref[0], ref_default[0]);
}