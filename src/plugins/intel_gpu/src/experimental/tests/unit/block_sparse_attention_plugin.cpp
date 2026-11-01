// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin-level (ov::Core) end-to-end test for the experimental BlockSparseAttention op on the
// Intel GPU plugin. Exercises the op graph through the public inference API (compile_model /
// create_infer_request / infer) and validates the output against the CPU reference
// (ov::reference::block_sparse_attention).
//
// Shape contract strictly follows the CM kernel hardware requirement: the sparse block size must
// be 128 or 256 (see cm_sdpa_common.hpp's sdpa_kernel_lsc_prefetch static_asserts). The mask tensors
// are constructed with an io::tensor 0/1 block mask; DraftBlockMask (the graph op that normally
// produces this mask in FlashVSR) is included/kept available for the same header/kernel surface.

#include "openvino/op/experimental/block_sparse_attention.hpp"
#include "openvino/op/experimental/draft_block_mask.hpp"
#include "openvino/reference/block_sparse_attention.hpp"
#include "openvino/reference/draft_block_mask.hpp"

#include <openvino/core/model.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test_utils.h"

using namespace ov::op;

namespace {

// Q/K/V layout is heads-inner, token-major [B, L, H, E]; block_mask is [B, H,
// ceil(L/block_size), ceil(L/block_size)]. block_size must be 128 or 256 to satisfy the CM kernel.
constexpr size_t kB = 1;
constexpr size_t kH = 4;
constexpr size_t kL = 256;
constexpr size_t kE = 64;
constexpr size_t kBlockSize = 128;
constexpr size_t kNumBlocks = kL / kBlockSize;  // 2

ov::float16 as_fp16(float val) {
    return ov::float16(static_cast<float>(val));
}

// Shared GPU-availability gate for the plugin-level tests: skips (or hard-fails when
// OV_EXP_REQUIRE_GPU=1) when no Intel GPU is present, so the tests can never be silently skipped.
bool require_gpu_or_skip(ov::Core& core) {
    const auto devices = core.get_available_devices();
    const bool has_gpu = std::find(devices.begin(), devices.end(), "GPU") != devices.end();
    const char* req_gpu = std::getenv("OV_EXP_REQUIRE_GPU");
    if (!has_gpu) {
        if (req_gpu && std::string(req_gpu) == "1") {
            ADD_FAILURE() << "Intel GPU required but not found";
        }
        return false;
    }
    return true;
}

}  // namespace

namespace ov::test {
class BlockSparseAttentionPluginTest : public ::testing::Test {};

TEST_F(BlockSparseAttentionPluginTest, CompileAndInferOnGpu) {
    ov::Core core;
    if (!require_gpu_or_skip(core)) {
        GTEST_SKIP() << "Intel GPU not found";
        return;
    }

    // ---- Build the model: q/k/v/mask -> BlockSparseAttention -> result ----
    const ov::Shape kv_shape{kB, kL, kH, kE};
    const ov::Shape mask_shape{kB, kH, kNumBlocks, kNumBlocks};

    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape(mask_shape));

    auto bsa = std::make_shared<ov::op::experimental::BlockSparseAttention>(q, k, v, mask,
                                                                             static_cast<int64_t>(kBlockSize),
                                                                             static_cast<int64_t>(kBlockSize),
                                                                             false);
    auto result = std::make_shared<ov::op::v0::Result>(bsa);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{q, k, v, mask});

    // ---- Compile and infer ----
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    // ---- Fill inputs with valid fp16 data ----
    auto q_tensor = request.get_input_tensor(0);
    auto k_tensor = request.get_input_tensor(1);
    auto v_tensor = request.get_input_tensor(2);

    const size_t qkv_count = kB * kL * kH * kE;
    std::vector<ov::float16> q_data(qkv_count);
    std::vector<ov::float16> k_data(qkv_count);
    std::vector<ov::float16> v_data(qkv_count);
    for (size_t i = 0; i < qkv_count; ++i) {
        const float val = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.05f;
        q_data[i] = as_fp16(val);
        k_data[i] = as_fp16(val);
        v_data[i] = as_fp16(val);
    }
    q_tensor.set_shape({kB, kL, kH, kE});
    k_tensor.set_shape({kB, kL, kH, kE});
    v_tensor.set_shape({kB, kL, kH, kE});
    std::copy(q_data.begin(), q_data.end(), q_tensor.data<ov::float16>());
    std::copy(k_data.begin(), k_data.end(), k_tensor.data<ov::float16>());
    std::copy(v_data.begin(), v_data.end(), v_tensor.data<ov::float16>());

    // Dense all-attend block mask.
    auto mask_tensor = request.get_input_tensor(3);
    mask_tensor.set_shape(mask_shape);
    std::vector<uint8_t> mask_data(kB * kH * kNumBlocks * kNumBlocks, 1);
    std::copy(mask_data.begin(), mask_data.end(), mask_tensor.data<uint8_t>());

    request.infer();

    // ---- Validate output shape and finiteness ----
    const ov::Tensor& out = request.get_output_tensor(0);
    EXPECT_EQ(out.get_shape(), ov::Shape({kB, kL, kH, kE}));
    const auto* out_ptr = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i) {
        const float val = static_cast<float>(out_ptr[i]);
        EXPECT_TRUE(std::isfinite(val)) << " non-finite output at index=" << i;
    }

    // ---- Compare against CPU reference ----
    std::vector<char> mask_ref(mask_data.begin(), mask_data.end());
    std::vector<ov::float16> ref_output(kB * kL * kH * kE);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kE));
    ov::reference::block_sparse_attention<ov::float16>(q_data.data(),
                                                       k_data.data(),
                                                       v_data.data(),
                                                       mask_ref.data(),
                                                       ref_output.data(),
                                                       kB,
                                                       kL,        // Lq
                                                       kL,        // Lk
                                                       kH,        // Hq
                                                       kH,        // Hkv
                                                       kE,        // E
                                                       kE,        // Ev
                                                       kB,        // Bm
                                                       kH,        // Hm
                                                       kBlockSize,
                                                       kBlockSize,
                                                       scale,
                                                       false);  // causal

    EXPECT_EQ(out.get_size(), ref_output.size());
    for (size_t i = 0; i < ref_output.size(); ++i) {
        const float actual = static_cast<float>(out_ptr[i]);
        const float expected = static_cast<float>(ref_output[i]);
        EXPECT_NEAR(actual, expected, 1e-2f) << " at index=" << i;
    }
}

// End-to-end cascade: DraftBlockMask dynamically derives a block-sparse mask from windowed
// query/key feature maps and feeds it directly into BlockSparseAttention's block_mask input. This
// exercises the real FlashVSR-style pipeline (draft path produces the top-k mask that the attended
// path consumes) through the public inference API on the GPU.
TEST_F(BlockSparseAttentionPluginTest, DraftBlockMaskToBlockSparseAttentionCascade) {
    ov::Core core;
    if (!require_gpu_or_skip(core)) {
        GTEST_SKIP() << "Intel GPU not found";
        return;
    }

    // DraftBlockMask geometry. Nq == seqlen * HW and Nk % HW == 0 must hold; output shape is
    // [batch_size, num_heads, Nq, Nk] = [1, 4, 2, 2], which is exactly the block_mask shape
    // BlockSparseAttention expects for L=256 / block_size=128.
    constexpr size_t kNq = 2;
    constexpr size_t kNk = 2;
    constexpr size_t kWindowTokens = kBlockSize;  // 128
    constexpr size_t kDraftD = 64;                // divisible by kH
    constexpr size_t kHW = 1;                     // local_mask square extent [HW, HW]
    constexpr size_t kSeqlen = kNumBlocks;        // 2 == kNq / kHW
    constexpr size_t kTopk = 1;

    // ---- Build the model: q/k feature maps -> DraftBlockMask -> BSA -> result ----
    const ov::Shape draft_q_shape{kNq, kWindowTokens, kDraftD};
    const ov::Shape draft_k_shape{kNk, kWindowTokens, kDraftD};
    const ov::Shape local_mask_shape{kHW, kHW};

    auto q_windows = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(draft_q_shape));
    auto k_windows = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(draft_k_shape));
    auto local_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape(local_mask_shape));

    auto draft = std::make_shared<ov::op::experimental::DraftBlockMask>(q_windows,
                                                                        k_windows,
                                                                        local_mask,
                                                                        static_cast<int64_t>(kH),
                                                                        static_cast<int64_t>(kSeqlen),
                                                                        static_cast<int64_t>(kTopk),
                                                                        static_cast<int64_t>(kB));

    const ov::Shape kv_shape{kB, kL, kH, kE};
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape(kv_shape));

    auto bsa = std::make_shared<ov::op::experimental::BlockSparseAttention>(q,
                                                                            k,
                                                                            v,
                                                                            draft,
                                                                            static_cast<int64_t>(kBlockSize),
                                                                            static_cast<int64_t>(kBlockSize),
                                                                            false);
    auto result = std::make_shared<ov::op::v0::Result>(bsa);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{q_windows, k_windows, local_mask, q, k, v});

    // ---- Compile and infer ----
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto qw_tensor = request.get_input_tensor(0);
    auto kw_tensor = request.get_input_tensor(1);
    auto lm_tensor = request.get_input_tensor(2);

    const size_t draft_count = kNq * kWindowTokens * kDraftD;
    std::vector<ov::float16> qw_data(draft_count);
    std::vector<ov::float16> kw_data(draft_count);
    for (size_t i = 0; i < draft_count; ++i) {
        const float val = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.05f;
        qw_data[i] = as_fp16(val);
        kw_data[i] = as_fp16(val);
    }
    qw_tensor.set_shape(draft_q_shape);
    kw_tensor.set_shape(draft_k_shape);
    std::copy(qw_data.begin(), qw_data.end(), qw_tensor.data<ov::float16>());
    std::copy(kw_data.begin(), kw_data.end(), kw_tensor.data<ov::float16>());

    // All-pass static local-window bias: the draft op still ranks key blocks by similarity.
    lm_tensor.set_shape(local_mask_shape);
    std::vector<uint8_t> local_mask_data(kHW * kHW, 1);
    std::copy(local_mask_data.begin(), local_mask_data.end(), lm_tensor.data<uint8_t>());

    auto q_tensor = request.get_input_tensor(3);
    auto k_tensor = request.get_input_tensor(4);
    auto v_tensor = request.get_input_tensor(5);

    const size_t qkv_count = kB * kL * kH * kE;
    std::vector<ov::float16> q_data(qkv_count);
    std::vector<ov::float16> k_data(qkv_count);
    std::vector<ov::float16> v_data(qkv_count);
    for (size_t i = 0; i < qkv_count; ++i) {
        const float val = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.05f;
        q_data[i] = as_fp16(val);
        k_data[i] = as_fp16(val);
        v_data[i] = as_fp16(val);
    }
    q_tensor.set_shape({kB, kL, kH, kE});
    k_tensor.set_shape({kB, kL, kH, kE});
    v_tensor.set_shape({kB, kL, kH, kE});
    std::copy(q_data.begin(), q_data.end(), q_tensor.data<ov::float16>());
    std::copy(k_data.begin(), k_data.end(), k_tensor.data<ov::float16>());
    std::copy(v_data.begin(), v_data.end(), v_tensor.data<ov::float16>());

    request.infer();

    // ---- Validate output shape and finiteness ----
    const ov::Tensor& out = request.get_output_tensor(0);
    EXPECT_EQ(out.get_shape(), ov::Shape({kB, kL, kH, kE}));
    const auto* out_ptr = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i) {
        const float val = static_cast<float>(out_ptr[i]);
        EXPECT_TRUE(std::isfinite(val)) << " non-finite output at index=" << i;
    }

    // ---- Compare against the CPU reference (draft mask -> block-sparse attention) ----
    const size_t mask_count = kB * kH * kNq * kNk;
    std::vector<char> mask_ref(mask_count);
    std::vector<char> local_mask_ref(local_mask_data.begin(), local_mask_data.end());
    ov::reference::draft_block_mask<ov::float16>(qw_data.data(),
                                                 kw_data.data(),
                                                 local_mask_ref.data(),
                                                 mask_ref.data(),
                                                 kNq,
                                                 kNk,
                                                 kWindowTokens,
                                                 kDraftD,
                                                 kH,
                                                 kHW,
                                                 kSeqlen,
                                                 kTopk,
                                                 kB);

    std::vector<ov::float16> ref_output(kB * kL * kH * kE);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kE));
    ov::reference::block_sparse_attention<ov::float16>(q_data.data(),
                                                       k_data.data(),
                                                       v_data.data(),
                                                       mask_ref.data(),
                                                       ref_output.data(),
                                                       kB,
                                                       kL,
                                                       kL,
                                                       kH,
                                                       kH,
                                                       kE,
                                                       kE,
                                                       kB,
                                                       kH,
                                                       kBlockSize,
                                                       kBlockSize,
                                                       scale,
                                                       false);

    EXPECT_EQ(out.get_size(), ref_output.size());
    for (size_t i = 0; i < ref_output.size(); ++i) {
        const float actual = static_cast<float>(out_ptr[i]);
        const float expected = static_cast<float>(ref_output[i]);
        EXPECT_NEAR(actual, expected, 1e-2f) << " at index=" << i;
    }
}

// Dynamic-shape variant: a dynamic token-count axis (PartialShape{1, -1, 4, 64}) forces the model
// through the requires_new_shape_infer() dynamic path in execution_config.cpp, validating that the
// experimental ops survive compile-time dynamic shapes and infer with a runtime-resolved shape on
// the GPU.
TEST_F(BlockSparseAttentionPluginTest, CompileAndInferDynamicDimensionOnGpu) {
    ov::Core core;
    if (!require_gpu_or_skip(core)) {
        GTEST_SKIP() << "Intel GPU not found";
        return;
    }

    // ---- Build a model with a dynamic L (token-count) axis ----
    auto dyn_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{kB, ov::Dimension::dynamic(), kH, kE});
    auto dyn_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{kB, ov::Dimension::dynamic(), kH, kE});
    auto dyn_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{kB, ov::Dimension::dynamic(), kH, kE});
    const ov::Shape mask_shape{kB, kH, kNumBlocks, kNumBlocks};
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape(mask_shape));

    auto bsa = std::make_shared<ov::op::experimental::BlockSparseAttention>(dyn_q,
                                                                            dyn_k,
                                                                            dyn_v,
                                                                            mask,
                                                                            static_cast<int64_t>(kBlockSize),
                                                                            static_cast<int64_t>(kBlockSize),
                                                                            false);
    auto result = std::make_shared<ov::op::v0::Result>(bsa);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{dyn_q, dyn_k, dyn_v, mask});

    // ---- Compile with dynamic shapes, then infer with a concrete runtime length ----
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto q_tensor = request.get_input_tensor(0);
    auto k_tensor = request.get_input_tensor(1);
    auto v_tensor = request.get_input_tensor(2);

    const size_t qkv_count = kB * kL * kH * kE;
    std::vector<ov::float16> q_data(qkv_count);
    std::vector<ov::float16> k_data(qkv_count);
    std::vector<ov::float16> v_data(qkv_count);
    for (size_t i = 0; i < qkv_count; ++i) {
        const float val = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.05f;
        q_data[i] = as_fp16(val);
        k_data[i] = as_fp16(val);
        v_data[i] = as_fp16(val);
    }
    q_tensor.set_shape({kB, kL, kH, kE});
    k_tensor.set_shape({kB, kL, kH, kE});
    v_tensor.set_shape({kB, kL, kH, kE});
    std::copy(q_data.begin(), q_data.end(), q_tensor.data<ov::float16>());
    std::copy(k_data.begin(), k_data.end(), k_tensor.data<ov::float16>());
    std::copy(v_data.begin(), v_data.end(), v_tensor.data<ov::float16>());

    auto mask_tensor = request.get_input_tensor(3);
    mask_tensor.set_shape(mask_shape);
    std::vector<uint8_t> mask_data(kB * kH * kNumBlocks * kNumBlocks, 1);
    std::copy(mask_data.begin(), mask_data.end(), mask_tensor.data<uint8_t>());

    request.infer();

    // ---- Validate output shape and finiteness ----
    const ov::Tensor& out = request.get_output_tensor(0);
    EXPECT_EQ(out.get_shape(), ov::Shape({kB, kL, kH, kE}));
    const auto* out_ptr = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i) {
        const float val = static_cast<float>(out_ptr[i]);
        EXPECT_TRUE(std::isfinite(val)) << " non-finite output at index=" << i;
    }
}

}  // namespace ov::test