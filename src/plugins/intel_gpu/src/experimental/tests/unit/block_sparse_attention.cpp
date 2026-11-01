// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/block_sparse_attention.hpp"

#include <cmath>
#include <cstddef>
#include <experimental/primitives/block_sparse_attention.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <vector>

#include "block_sparse_attention_inst.h"
#include "experimental/cm_db/kernels_db.hpp"
#include "graph/impls/ocl/kernel_selector_helper.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

bool is_cm_jit_and_immad_supported() {
    try {
        auto& engine = get_test_engine();
        auto config = get_test_default_config(engine);
        return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
    } catch (...) {
        return false;
    }
}

// GPU-side implementation requires fp16 Q/K/V (see BlockSparseAttentionImplementationManager::
// validate_impl); the CPU reference is templated, so it's exercised here with the same precision
// for a fair comparison rather than introducing extra fp32->fp16 rounding noise.
struct BlockSparseAttentionTestParams {
    size_t B = 1;
    size_t Lq = 128;
    size_t Lk = 128;
    size_t Hq = 2;
    size_t Hkv = 2;
    size_t E = 64;
    size_t Ev = 64;
    size_t block_size_q = 128;
    size_t block_size_kv = 128;
    bool causal = false;
    std::vector<uint8_t> block_mask;  // [B, Hq, num_q_blocks, num_k_blocks], empty => all-ones

    // Which axes are declared *dynamic* (ov::Dimension::dynamic()) on the compiled topology's
    // input_layout, independent of the concrete sizes above (which are always used for the actual
    // allocated memory and the CPU reference). All-false (the default) reproduces the fully-static
    // topology every pre-existing case below already used. `dyn_tokens` covers the axis this whole
    // fix is about (query/key/value's L axis and the mask's two block-count axes) -- see
    // resolve_geometry() in block_sparse_attention.cpp. `dyn_batch`/`dyn_num_heads`/`dyn_head_size`
    // cover axes resolve_geometry() still requires to be static.
    bool dyn_batch = false;
    bool dyn_num_heads = false;         // query's Hq axis and key/value's Hkv axis
    bool dyn_head_size = false;         // query/key/value's E/Ev axis
    bool dyn_tokens = false;            // query/key/value's L axis, and (unless overridden below) the
                                        // mask's block-count axes
    bool dyn_mask_blocks_only = false;  // mask's block-count axes dynamic while Q/K/V's L axis stays
                                        // static -- isolates the mask-only case from `dyn_tokens`
};

ov::Dimension maybe_dynamic(bool dynamic, int64_t value) {
    return dynamic ? ov::Dimension::dynamic() : ov::Dimension(value);
}

// Builds the topology (with layouts reflecting `p`'s `dyn_*` flags) and, if `expect_throw` is
// false, executes it once with `p`'s concrete sizes and checks the output against the CPU
// reference. Returns the constructed network so multi-execution tests can drive it further.
network::ptr build_and_run_block_sparse_attention(const std::string& seed, BlockSparseAttentionTestParams p, bool skip_run = false) {
    auto& engine = get_test_engine();

    const size_t num_q_blocks = (p.Lq + p.block_size_q - 1) / p.block_size_q;
    const size_t num_k_blocks = (p.Lk + p.block_size_kv - 1) / p.block_size_kv;

    tests::random_generator rg;
    rg.set_seed(seed);

    auto q_data = rg.generate_random_1d<ov::float16>(p.B * p.Lq * p.Hq * p.E, -1, 1, 256);
    auto k_data = rg.generate_random_1d<ov::float16>(p.B * p.Lk * p.Hkv * p.E, -1, 1, 256);
    auto v_data = rg.generate_random_1d<ov::float16>(p.B * p.Lk * p.Hkv * p.Ev, -1, 1, 256);

    if (p.block_mask.empty()) {
        p.block_mask.assign(p.B * p.Hq * num_q_blocks * num_k_blocks, 1);
    }
    std::vector<char> mask_ref(p.block_mask.begin(), p.block_mask.end());

    auto q_dyn_layout = layout{ov::PartialShape{maybe_dynamic(p.dyn_batch, p.B),
                                                maybe_dynamic(p.dyn_tokens, p.Lq),
                                                maybe_dynamic(p.dyn_num_heads, p.Hq),
                                                maybe_dynamic(p.dyn_head_size, p.E)},
                               data_types::f16,
                               format::bfyx};
    auto k_dyn_layout = layout{ov::PartialShape{maybe_dynamic(p.dyn_batch, p.B),
                                                maybe_dynamic(p.dyn_tokens, p.Lk),
                                                maybe_dynamic(p.dyn_num_heads, p.Hkv),
                                                maybe_dynamic(p.dyn_head_size, p.E)},
                               data_types::f16,
                               format::bfyx};
    auto v_dyn_layout = layout{ov::PartialShape{maybe_dynamic(p.dyn_batch, p.B),
                                                maybe_dynamic(p.dyn_tokens, p.Lk),
                                                maybe_dynamic(p.dyn_num_heads, p.Hkv),
                                                maybe_dynamic(p.dyn_head_size, p.Ev)},
                               data_types::f16,
                               format::bfyx};
    const bool dyn_mask_blocks = p.dyn_tokens || p.dyn_mask_blocks_only;
    auto mask_dyn_layout = layout{ov::PartialShape{static_cast<int64_t>(p.B),
                                                   static_cast<int64_t>(p.Hq),
                                                   maybe_dynamic(dyn_mask_blocks, static_cast<int64_t>(num_q_blocks)),
                                                   maybe_dynamic(dyn_mask_blocks, static_cast<int64_t>(num_k_blocks))},
                                  data_types::u8,
                                  format::bfyx};

    auto q_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lq), static_cast<int64_t>(p.Hq), static_cast<int64_t>(p.E)},
               data_types::f16,
               format::bfyx};
    auto k_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.E)},
               data_types::f16,
               format::bfyx};
    auto v_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.Ev)},
               data_types::f16,
               format::bfyx};
    auto mask_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Hq), static_cast<int64_t>(num_q_blocks), static_cast<int64_t>(num_k_blocks)},
               data_types::u8,
               format::bfyx};

    auto q_mem = engine.allocate_memory(q_actual_layout);
    auto k_mem = engine.allocate_memory(k_actual_layout);
    auto v_mem = engine.allocate_memory(v_actual_layout);
    auto mask_mem = engine.allocate_memory(mask_actual_layout);

    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(mask_mem, p.block_mask);

    topology topo;
    topo.add(input_layout("query", q_dyn_layout));
    topo.add(input_layout("key", k_dyn_layout));
    topo.add(input_layout("value", v_dyn_layout));
    topo.add(input_layout("block_mask", mask_dyn_layout));
    topo.add(block_sparse_attention("block_sparse_attention",
                                    {input_info("query"), input_info("key"), input_info("value"), input_info("block_mask")},
                                    static_cast<int64_t>(p.block_size_q),
                                    static_cast<int64_t>(p.block_size_kv),
                                    p.causal));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    if (skip_run) {
        return network;
    }

    network->set_input_data("query", q_mem);
    network->set_input_data("key", k_mem);
    network->set_input_data("value", v_mem);
    network->set_input_data("block_mask", mask_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("block_sparse_attention").get_memory();

    std::vector<ov::float16> ref_output(p.B * p.Lq * p.Hq * p.Ev);
    const float scale = 1.0f / std::sqrt(static_cast<float>(p.E));
    ov::reference::block_sparse_attention<ov::float16>(q_data.data(),
                                                       k_data.data(),
                                                       v_data.data(),
                                                       mask_ref.data(),
                                                       ref_output.data(),
                                                       p.B,
                                                       p.Lq,
                                                       p.Lk,
                                                       p.Hq,
                                                       p.Hkv,
                                                       p.E,
                                                       p.Ev,
                                                       p.B,
                                                       p.Hq,
                                                       p.block_size_q,
                                                       p.block_size_kv,
                                                       scale,
                                                       p.causal);

    EXPECT_EQ(output_mem->count(), ref_output.size());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
    for (size_t i = 0; i < ref_output.size(); i++) {
        const float actual = static_cast<float>(out_ptr[i]);
        const float expected = static_cast<float>(ref_output[i]);
        const float tol = std::max(2e-2f, std::abs(expected) * 2e-2f);
        EXPECT_NEAR(actual, expected, tol) << " at index=" << i;
    }

    return network;
}

void run_block_sparse_attention_test(const std::string& seed, BlockSparseAttentionTestParams p) {
    build_and_run_block_sparse_attention(seed, std::move(p));
}

TEST(block_sparse_attention_gpu, prefill_noncausal_dense) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    run_block_sparse_attention_test("block_sparse_attention_prefill_noncausal_dense", p);
}

TEST(block_sparse_attention_gpu, prefill_causal_dense) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = true;
    run_block_sparse_attention_test("block_sparse_attention_prefill_causal_dense", p);
}

TEST(block_sparse_attention_gpu, prefill_gqa) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 4;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    run_block_sparse_attention_test("block_sparse_attention_prefill_gqa", p);
}

TEST(block_sparse_attention_gpu, prefill_sparse_block) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 256;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    // 1 q-block x 2 kv-blocks per head; skip the 2nd kv block for every head.
    p.block_mask = {1,
                    0,  // head 0: keep block0, skip block1
                    1,
                    0};  // head 1: keep block0, skip block1
    run_block_sparse_attention_test("block_sparse_attention_prefill_sparse_block", p);
}

// DraftBlockMask performs a global (head, temporal-group) top-k, so a query block can end up with
// every kv block skipped (the local mask is only a pre-softmax bias, not a survival guarantee
// after top-k) -- cur_sum for such a row would be exactly 0 without cm_sdpa_common.hpp's
// cur_sum_is_zero guard, producing NaN (cm_inv(0) == +Inf times a zero rO accumulator) instead of
// the CPU reference's 0 (ov::reference::block_sparse_attention's `sum > 0.0f ? 1.0f/sum : 0.0f`).
TEST(block_sparse_attention_gpu, prefill_all_masked_row) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 256;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    // 1 q-block x 2 kv-blocks per head; skip every kv block for every head.
    p.block_mask = {0, 0, 0, 0};
    run_block_sparse_attention_test("block_sparse_attention_prefill_all_masked_row", p);
}

// Single-token (Lq==1) is normalized onto the block-sparse prefill kernel (there is no separate
// decode path anymore). This exercises the Lq==1 + non-all-ones block_mask case: the sparse mask
// must actually be honored (not a silent dense fallback), exactly matching the CPU reference.
TEST(block_sparse_attention_gpu, prefill_sparse_block_single_query) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 1;
    p.Lk = 300;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    // 1 q-block x 3 kv-blocks per head; skip the middle kv block (block index 1) for every head.
    // num_k_blocks = ceil(300/128) = 3, so the mask is [B=1, Hq=2, 1, 3].
    p.block_mask = {1, 0, 1, 1, 0, 1};
    run_block_sparse_attention_test("block_sparse_attention_prefill_sparse_block_single_query", p);
}

// Single-token causal case: every block beyond the first is masked by causality. The reference
// is called with the same dense all-ones block_mask (the causal flag inside block_sparse_attention
// applies the causal triangle on top of the block mask), so agreement proves causality is honored
// even at Lq==1 on the prefill path.
TEST(block_sparse_attention_gpu, prefill_single_query_causal) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 1;
    p.Lk = 300;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = true;
    run_block_sparse_attention_test("block_sparse_attention_prefill_single_query_causal", p);
}

// ===================== dynamic-shape coverage (see resolve_geometry() in
// block_sparse_attention.cpp) =====================
//
// FlashVSR's streaming DiT graph gives this op a dynamic token-count axis (Lq/Lk) and,
// consequently, a dynamic mask block-count axis -- batch/num_heads/num_kv_heads/head_size stay
// compile-time constants. The cases below check that exact split: the axes resolve_geometry()
// treats as safe to leave dynamic actually work (including being re-dispatched for more than one
// concrete shape against the same compiled network), while the axes it still requires to be
// static are rejected rather than silently mis-dispatched.

TEST(block_sparse_attention_gpu, dynamic_tokens_prefill_dense) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    p.dyn_tokens = true;
    auto network = build_and_run_block_sparse_attention("block_sparse_attention_dynamic_tokens_prefill_dense", p);

    auto& node = network->get_program()->get_node("block_sparse_attention");
    auto impl = node.get_selected_impl();
    ASSERT_TRUE(impl != nullptr);
    EXPECT_TRUE(impl->is_dynamic());
}

TEST(block_sparse_attention_gpu, dynamic_tokens_prefill_sparse_block) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 256;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    p.dyn_tokens = true;
    // Same 1 q-block x 2 kv-blocks skip pattern as prefill_sparse_block above.
    p.block_mask = {1, 0, 1, 0};
    build_and_run_block_sparse_attention("block_sparse_attention_dynamic_tokens_prefill_sparse_block", p);
}

TEST(block_sparse_attention_gpu, dynamic_tokens_causal) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = true;
    p.dyn_tokens = true;
    build_and_run_block_sparse_attention("block_sparse_attention_dynamic_tokens_causal", p);
}

TEST(block_sparse_attention_gpu, dynamic_mask_blocks_only_dense) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    // Mirrors FlashVSR's mask-building side, where the mask's block-count axes can be dynamic
    // even in isolation (see build_flashvsr_block_attention_mask_dynamic_kv) -- Q/K/V's L axis is
    // kept static here to isolate this from dynamic_tokens_* above.
    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    p.dyn_mask_blocks_only = true;
    build_and_run_block_sparse_attention("block_sparse_attention_dynamic_mask_blocks_only_dense", p);
}

// Same compiled network (one dynamic-shape `block_sparse_attention` node), driven through two
// `execute()` calls with genuinely different concrete Lq/Lk -- the scenario that used to crash
// during compile_model() before resolve_geometry() stopped requiring Lq/Lk to be static. Verifies
// get_dispatch_data_func() is correctly re-invoked (not stale from the first shape) for the second
// shape, per PrimitiveImplOCL::execute_stage's need_dispatch_data_update handling.
TEST(block_sparse_attention_gpu, dynamic_tokens_multi_inference_different_shape) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    auto& engine = get_test_engine();

    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;
    p.dyn_tokens = true;

    // Build the network once against the dynamic-token declared layout; p.Lq/p.Lk here only
    // affect the (unused, skip_run=true) actual-layout allocation, not the declared shape.
    p.Lq = 128;
    p.Lk = 128;
    auto network = build_and_run_block_sparse_attention("block_sparse_attention_dynamic_tokens_multi_shape_build", p, /*skip_run=*/true);

    auto run_shape = [&](const std::string& seed, size_t Lq, size_t Lk) {
        tests::random_generator rg;
        rg.set_seed(seed);
        const size_t num_q_blocks = (Lq + p.block_size_q - 1) / p.block_size_q;
        const size_t num_k_blocks = (Lk + p.block_size_kv - 1) / p.block_size_kv;

        auto q_data = rg.generate_random_1d<ov::float16>(p.B * Lq * p.Hq * p.E, -1, 1, 256);
        auto k_data = rg.generate_random_1d<ov::float16>(p.B * Lk * p.Hkv * p.E, -1, 1, 256);
        auto v_data = rg.generate_random_1d<ov::float16>(p.B * Lk * p.Hkv * p.Ev, -1, 1, 256);
        std::vector<uint8_t> mask_data(p.B * p.Hq * num_q_blocks * num_k_blocks, 1);
        std::vector<char> mask_ref(mask_data.begin(), mask_data.end());

        auto q_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(Lq), static_cast<int64_t>(p.Hq), static_cast<int64_t>(p.E)},
                   data_types::f16,
                   format::bfyx});
        auto k_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.E)},
                   data_types::f16,
                   format::bfyx});
        auto v_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.Ev)},
                   data_types::f16,
                   format::bfyx});
        auto mask_mem = engine.allocate_memory(layout{
            ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Hq), static_cast<int64_t>(num_q_blocks), static_cast<int64_t>(num_k_blocks)},
            data_types::u8,
            format::bfyx});
        set_values(q_mem, q_data);
        set_values(k_mem, k_data);
        set_values(v_mem, v_data);
        set_values(mask_mem, mask_data);

        network->set_input_data("query", q_mem);
        network->set_input_data("key", k_mem);
        network->set_input_data("value", v_mem);
        network->set_input_data("block_mask", mask_mem);

        auto outputs = network->execute();
        auto output_mem = outputs.at("block_sparse_attention").get_memory();

        std::vector<ov::float16> ref_output(p.B * Lq * p.Hq * p.Ev);
        const float scale = 1.0f / std::sqrt(static_cast<float>(p.E));
        ov::reference::block_sparse_attention<ov::float16>(q_data.data(),
                                                           k_data.data(),
                                                           v_data.data(),
                                                           mask_ref.data(),
                                                           ref_output.data(),
                                                           p.B,
                                                           Lq,
                                                           Lk,
                                                           p.Hq,
                                                           p.Hkv,
                                                           p.E,
                                                           p.Ev,
                                                           p.B,
                                                           p.Hq,
                                                           p.block_size_q,
                                                           p.block_size_kv,
                                                           scale,
                                                           p.causal);

        EXPECT_EQ(output_mem->count(), ref_output.size());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
        for (size_t i = 0; i < ref_output.size(); i++) {
            const float actual = static_cast<float>(out_ptr[i]);
            const float expected = static_cast<float>(ref_output[i]);
            const float tol = std::max(2e-2f, std::abs(expected) * 2e-2f);
            EXPECT_NEAR(actual, expected, tol) << " Lq=" << Lq << " Lk=" << Lk << " at index=" << i;
        }
    };

    run_shape("block_sparse_attention_dynamic_tokens_multi_shape_1", 128, 128);
    run_shape("block_sparse_attention_dynamic_tokens_multi_shape_2", 256, 256);
    run_shape("block_sparse_attention_dynamic_tokens_multi_shape_3", 128, 384);
}

// The axes resolve_geometry() still requires to be static (batch/num_heads/num_kv_heads/
// head_size): declaring any of them dynamic must not silently mis-dispatch. Today that surfaces
// as an exception out of get_network() (constructing/compiling the program): the CM kernel's
// get_jit_constants() reads exactly these fields, throws via resolve_geometry()'s
// OPENVINO_ASSERT, and -- unlike a genuinely-missing-impl case -- there is no fallback
// implementation for block_sparse_attention to silently degrade to (see
// Registry<block_sparse_attention>::get_implementations(), CM-only).
// ===================== runtime scale (5-input variant) coverage =====================
//
// Exercises the primitive's optional 5th (`scale`) input, read on-device at dispatch time by
// read_scale_value() -- see block_sparse_attention_inst.h's get_shape_infer_dependencies(), which
// must list the scale input's index for kernel_impl_params::memory_deps to carry it.
//
// Declares query/key/value/mask dynamic (like dynamic_tokens_prefill_dense above): memory_deps is
// only populated by primitive_inst::update_shape()'s runtime-shape-refresh path
// (primitive_inst.cpp:397-471), which early-returns for a fully-static node whose output layout
// hasn't changed (:421-422) -- exactly like DraftBlockMask's own runtime-topk variant, which is
// likewise only ever exercised together with dyn_tokens (see
// dynamic_tokens_with_runtime_topk/runtime_topk_input_matches_attribute_topk in
// draft_block_mask.cpp). This mirrors FlashVSR's actual usage: the runtime scale/topk inputs exist
// specifically for the dynamic-shape dit graph.
TEST(block_sparse_attention_gpu, dynamic_tokens_with_runtime_scale) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    auto& engine = get_test_engine();

    BlockSparseAttentionTestParams p;
    p.B = 1;
    p.Lq = 128;
    p.Lk = 128;
    p.Hq = 2;
    p.Hkv = 2;
    p.E = 64;
    p.Ev = 64;
    p.block_size_q = 128;
    p.block_size_kv = 128;
    p.causal = false;

    const size_t num_q_blocks = (p.Lq + p.block_size_q - 1) / p.block_size_q;
    const size_t num_k_blocks = (p.Lk + p.block_size_kv - 1) / p.block_size_kv;

    tests::random_generator rg;
    rg.set_seed("block_sparse_attention_runtime_scale_prefill");

    auto q_data = rg.generate_random_1d<ov::float16>(p.B * p.Lq * p.Hq * p.E, -1, 1, 256);
    auto k_data = rg.generate_random_1d<ov::float16>(p.B * p.Lk * p.Hkv * p.E, -1, 1, 256);
    auto v_data = rg.generate_random_1d<ov::float16>(p.B * p.Lk * p.Hkv * p.Ev, -1, 1, 256);
    std::vector<uint8_t> mask_data(p.B * p.Hq * num_q_blocks * num_k_blocks, 1);
    std::vector<char> mask_ref(mask_data.begin(), mask_data.end());

    // Deliberately different from the default 1/sqrt(E) scale, so the test only passes if the
    // runtime value is actually read (and not the CM kernel's own default).
    const float runtime_scale = 0.5f / std::sqrt(static_cast<float>(p.E));
    std::vector<ov::float16> scale_data{ov::float16(runtime_scale)};

    auto q_mem = engine.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lq), static_cast<int64_t>(p.Hq), static_cast<int64_t>(p.E)},
               data_types::f16,
               format::bfyx});
    auto k_mem = engine.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.E)},
               data_types::f16,
               format::bfyx});
    auto v_mem = engine.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Lk), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.Ev)},
               data_types::f16,
               format::bfyx});
    auto mask_mem = engine.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Hq), static_cast<int64_t>(num_q_blocks), static_cast<int64_t>(num_k_blocks)},
               data_types::u8,
               format::bfyx});
    auto scale_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::f16, format::bfyx});

    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(mask_mem, mask_data);
    set_values(scale_mem, scale_data);

    // Dynamic Lq/Lk (and, consequently, the mask's block-count axes) -- see dyn_tokens in
    // BlockSparseAttentionTestParams above; batch/num_heads/head_size stay static.
    auto q_dyn_layout = layout{ov::PartialShape{static_cast<int64_t>(p.B), ov::Dimension::dynamic(), static_cast<int64_t>(p.Hq), static_cast<int64_t>(p.E)},
                               data_types::f16,
                               format::bfyx};
    auto k_dyn_layout = layout{ov::PartialShape{static_cast<int64_t>(p.B), ov::Dimension::dynamic(), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.E)},
                               data_types::f16,
                               format::bfyx};
    auto v_dyn_layout = layout{ov::PartialShape{static_cast<int64_t>(p.B), ov::Dimension::dynamic(), static_cast<int64_t>(p.Hkv), static_cast<int64_t>(p.Ev)},
                               data_types::f16,
                               format::bfyx};
    auto mask_dyn_layout = layout{ov::PartialShape{static_cast<int64_t>(p.B), static_cast<int64_t>(p.Hq), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                  data_types::u8,
                                  format::bfyx};

    topology topo;
    topo.add(input_layout("query", q_dyn_layout));
    topo.add(input_layout("key", k_dyn_layout));
    topo.add(input_layout("value", v_dyn_layout));
    topo.add(input_layout("block_mask", mask_dyn_layout));
    topo.add(input_layout("scale", scale_mem->get_layout()));
    topo.add(block_sparse_attention("block_sparse_attention",
                                    {input_info("query"), input_info("key"), input_info("value"), input_info("block_mask"), input_info("scale")},
                                    static_cast<int64_t>(p.block_size_q),
                                    static_cast<int64_t>(p.block_size_kv),
                                    p.causal));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("query", q_mem);
    network->set_input_data("key", k_mem);
    network->set_input_data("value", v_mem);
    network->set_input_data("block_mask", mask_mem);
    network->set_input_data("scale", scale_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("block_sparse_attention").get_memory();

    std::vector<ov::float16> ref_output(p.B * p.Lq * p.Hq * p.Ev);
    ov::reference::block_sparse_attention<ov::float16>(q_data.data(),
                                                       k_data.data(),
                                                       v_data.data(),
                                                       mask_ref.data(),
                                                       ref_output.data(),
                                                       p.B,
                                                       p.Lq,
                                                       p.Lk,
                                                       p.Hq,
                                                       p.Hkv,
                                                       p.E,
                                                       p.Ev,
                                                       p.B,
                                                       p.Hq,
                                                       p.block_size_q,
                                                       p.block_size_kv,
                                                       runtime_scale,
                                                       p.causal);

    EXPECT_EQ(output_mem->count(), ref_output.size());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
    for (size_t i = 0; i < ref_output.size(); i++) {
        const float actual = static_cast<float>(out_ptr[i]);
        const float expected = static_cast<float>(ref_output[i]);
        const float tol = std::max(2e-2f, std::abs(expected) * 2e-2f);
        EXPECT_NEAR(actual, expected, tol) << " at index=" << i;
    }
}

TEST(block_sparse_attention_gpu, dynamic_batch_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.dyn_batch = true;
    ASSERT_ANY_THROW(build_and_run_block_sparse_attention("block_sparse_attention_dynamic_batch_rejected", p));
}

TEST(block_sparse_attention_gpu, dynamic_num_heads_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.dyn_num_heads = true;
    ASSERT_ANY_THROW(build_and_run_block_sparse_attention("block_sparse_attention_dynamic_num_heads_rejected", p));
}

TEST(block_sparse_attention_gpu, dynamic_head_size_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.dyn_head_size = true;
    ASSERT_ANY_THROW(build_and_run_block_sparse_attention("block_sparse_attention_dynamic_head_size_rejected", p));
}

// The prefill kernel's DPAS REG_K == 16 block tiling and LSC 2D-Block descriptor width/pitch
// alignment require head_size (E) to be a multiple of 16, so a head_size that isn't a multiple of
// REG_K would issue mis-aligned accesses that overflow past the query/output buffer's head-size
// stride -- validate_impl rejects it up front (see
// BlockSparseAttentionImplementationManager::validate_impl) rather than letting the CM impl run
// with a mis-strided head_size. Statically-shaped (not the dyn_head_size case above) so this
// exercises the static-shape branch of that same guard.
TEST(block_sparse_attention_gpu, non_reg_k_multiple_head_size_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.E = 72;
    p.Ev = 72;
    ASSERT_ANY_THROW(build_and_run_block_sparse_attention("block_sparse_attention_non_reg_k_multiple_head_size_rejected", p));
}

// The ported kernel (block_sparse_attention_prefill.cm) uses a single CMFLA_HEAD_SIZE (= E,
// query/key's head size) throughout for output and scratch addressing, even though the core
// op/spec otherwise allow a distinct value head size (Ev != E) -- validate_impl rejects the
// mismatch up front (see BlockSparseAttentionImplementationManager::validate_impl) rather than
// letting the CM impl mis-address memory. Every other test in this file keeps E == Ev; this is
// the only one that doesn't.
TEST(block_sparse_attention_gpu, ev_ne_e_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    BlockSparseAttentionTestParams p;
    p.E = 64;
    p.Ev = 32;
    ASSERT_ANY_THROW(build_and_run_block_sparse_attention("block_sparse_attention_ev_ne_e_rejected", p));
}

// B10 regression tests: validate_impl rejects i8 mask (only u8 allowed) and non-fp16 scale.
TEST(block_sparse_attention_gpu, i8_mask_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    auto& engine = get_test_engine();
    topology topo;
    topo.add(input_layout("query", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("value", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("block_mask", layout{ov::PartialShape{1, 2, 1, 1}, data_types::i8, format::bfyx}));
    topo.add(block_sparse_attention("block_sparse_attention",
                                    {input_info("query"), input_info("key"), input_info("value"), input_info("block_mask")},
                                    128,
                                    128,
                                    false));
    auto config = get_test_default_config(engine);
    ASSERT_ANY_THROW(get_network(engine, topo, config, get_test_stream_ptr(), false));
}

TEST(block_sparse_attention_gpu, non_fp16_scale_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    auto& engine = get_test_engine();
    topology topo;
    topo.add(input_layout("query", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("value", layout{ov::PartialShape{1, 128, 2, 64}, data_types::f16, format::bfyx}));
    topo.add(input_layout("block_mask", layout{ov::PartialShape{1, 2, 1, 1}, data_types::u8, format::bfyx}));
    topo.add(input_layout("scale", layout{ov::PartialShape{}, data_types::f32, format::bfyx}));
    topo.add(block_sparse_attention("block_sparse_attention",
                                    {input_info("query"), input_info("key"), input_info("value"), input_info("block_mask"), input_info("scale")},
                                    128,
                                    128,
                                    false));
    auto config = get_test_default_config(engine);
    ASSERT_ANY_THROW(get_network(engine, topo, config, get_test_stream_ptr(), false));
}

#if defined(ENABLE_CM_FOR_GPU) && ENABLE_CM_FOR_GPU
// Build-independent Host test (no GPU / CM JIT required): the generated experimental CM kernel
// template for block_sparse_attention_prefill must contain the expected REG_M macro and include
// <cm/cm.h>, and must contain no leftover unresolved local `#include "..."` references. This guards
// against the codegen silently dropping or leaving an unexpanded header (see Phase 2 B2).
TEST(block_sparse_attention_gpu, kernel_template_codegen_complete) {
    auto template_str = std::string(ov::intel_gpu::experimental::cm::ExperimentalSourcesDB::get_kernel_template("block_sparse_attention_prefill"));
    ASSERT_FALSE(template_str.empty());
    ASSERT_NE(template_str.find("#define REG_M"), std::string::npos);
    ASSERT_TRUE(template_str.find("#include <cm/cm.h>") != std::string::npos || template_str.find("#include<cm/cm.h>") != std::string::npos);
    ASSERT_EQ(template_str.find("#include \""), std::string::npos);
}
#endif

}  // namespace
