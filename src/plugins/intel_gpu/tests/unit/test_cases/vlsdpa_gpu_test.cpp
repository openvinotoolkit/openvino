// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/util/file_util.hpp"

#include <intel_gpu/runtime/debug_configuration.hpp>
#include <intel_gpu/primitives/input_layout.hpp>


#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include <intel_gpu/primitives/vl_sdpa.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include "scaled_dot_product_attention_inst.h"
#include "vl_sdpa_inst.h"

#include <cstddef>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace cldnn;
using namespace ::tests;
namespace  {
struct vlsdpa_test_params {
    int32_t head_size;
    int32_t num_heads;
    std::vector<int32_t> cu_seqlens;

    vlsdpa_test_params(int h_size, int n_heads, std::vector<int32_t> _cu_seqlens)
        : head_size(h_size), num_heads(n_heads), cu_seqlens(_cu_seqlens) {}
};

struct vlsdpa_gpu_test : public ::testing::TestWithParam<vlsdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
        set_values(mem, input_data);
    }

    void get_attention_mask(cldnn::memory::ptr mem, std::vector<int32_t> cu_seqlens) {
        const auto shapes = mem->get_layout().get_shape();
        const size_t size = ov::shape_size(shapes);
        const auto hidden_states_size = shapes.back();

        std::vector<ov::float16> attention_mask_data(size, -std::numeric_limits<ov::float16>::infinity());
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            size_t start = cu_seqlens[i-1];
            size_t end = cu_seqlens[i];
            for (size_t row = start; row < end; ++row) {
                for (size_t col = start; col < end; ++col) {
                    attention_mask_data[row * hidden_states_size + col] = ov::float16(0.0f);
                }
            }
        }
        set_values(mem, attention_mask_data);
    }

    void get_cu_seqlens(cldnn::memory::ptr mem, std::vector<int32_t> cu_seqlens) {
        set_values(mem, cu_seqlens);
    }

    topology create_ref_topology(layout q_layout, layout k_layout, layout v_layout, layout attn_mask_layout) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("attn", attn_mask_layout));  // "attention_mask"

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            false, -1, order_q, order_k, order_v, {0, 1, 2}, {}, false);
        topo.add(sdpa_prim);

        topo.add(permute("permute_o", input_info("sdpa"), order_o2));

        topo.add(reorder("result",input_info("permute_o"), format::bfyx, data_types::f16));
        return topo;
    };

    topology create_topology(layout q_layout, layout k_layout, layout v_layout, layout cu_seqlen_layout) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("attn", cu_seqlen_layout));  // "cu_seqlens"

        auto vlsdpa_prim = vl_sdpa("vlsdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            order_q, order_k, order_v, order_o);

        topo.add(vlsdpa_prim);
        topo.add(reorder("result",input_info("vlsdpa"), format::bfyx, data_types::f16));
        return topo;
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(topology &topo,
            cldnn::memory::ptr q_mem,
            cldnn::memory::ptr k_mem,
            cldnn::memory::ptr v_mem,
            cldnn::memory::ptr attn_mem,
            const bool is_caching_test) {
        auto& engine = get_test_engine();

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("attn", attn_mem);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(vlsdpa_test_params& p, const bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto cu_seqlens = p.cu_seqlens;
        const auto cumsum = cu_seqlens.back();
        const auto seq_length_q = cumsum;
        const auto seq_length_kv = cumsum;

        assert(cu_seqlens.front() == 0);

        auto& engine = get_test_engine();

        // create topologies
        cldnn::layout q_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_dyn_layout({1, -1, -1}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_dyn_layout({-1}, data_types::i32, format::bfyx);

        topology ref_topo = create_ref_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, attn_mask_dyn_layout);
        topology opt_topo = create_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, cu_seqlens_dyn_layout);

        // allocate memories
        cldnn::layout q_static_layout({seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_static_layout({seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_static_layout({seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_static_layout({1, seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_static_layout({static_cast<ov::Dimension::value_type>(cu_seqlens.size())}, data_types::i32, format::bfyx);

        auto q_mem = engine.allocate_memory(q_static_layout);
        auto k_mem = engine.allocate_memory(k_static_layout);
        auto v_mem = engine.allocate_memory(v_static_layout);
        auto attn_mask_mem = engine.allocate_memory(attn_mask_static_layout);
        auto cu_seqlens_mem = engine.allocate_memory(cu_seqlens_static_layout);

        load_input(q_mem, 0);
        load_input(k_mem, 1);
        load_input(v_mem, 2);
        get_attention_mask(attn_mask_mem, cu_seqlens);
        get_cu_seqlens(cu_seqlens_mem, cu_seqlens);

        // execute networks
        auto [mem_ref_ptr, net_ref_ptr] = run_network(ref_topo,
                                        q_mem, k_mem, v_mem, attn_mask_mem,
                                        is_caching_test);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(opt_topo,
                                        q_mem, k_mem, v_mem, cu_seqlens_mem,
                                        is_caching_test);

        // validate results
        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        {
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<vlsdpa_test_params>& info) {
        std::string result = "vlsdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               vec2str<int32_t>(info.param.cu_seqlens);

        return result;
    }

    static bool check_vlsdpa_available() {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        if (!cldnn::check_cm_jit_support(engine, config) || !engine.get_device_info().supports_immad) {
            return false;
        }

        return true;
    }

    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_o = {1, 0, 2};
    const std::vector<uint16_t> order_o2 = {1, 0, 2};  // primitive permute constructor asks "ushort"
};

TEST_P(vlsdpa_gpu_test, basic) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p);
}

TEST_P(vlsdpa_gpu_test, basic_caching) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(smoke_vlsdpa_gpu_test,
    vlsdpa_gpu_test,
    ::testing::Values(
        vlsdpa_test_params{64 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{128 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 2 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 2 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 1 /*num_head*/, {0, 16, 32} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 2 /*num_head*/, {0, 16, 32} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 2 /*num_head*/, {0, 16, 32} /*cu_seqlens*/}
    ),
    vlsdpa_gpu_test::PrintToStringParamName
);

} // namespace

// ============================================================================
// Regression: vl_sdpa CM kernel uses wrong KV pointer offset and per-token
// pitch when Q/K/V are in-place crop aliases of a packed QKV [L, 3, H, S].
//
// Two compounding bugs (present without the fix):
//  1. token_offset_kv is derived from K's layout padding (slice=1, pad=1) and
//     applied identically to V.  V has slice=2 (pad=2), so its correct memory
//     offset is 2*H*S; it receives 1*H*S and reads K data instead of V data.
//  2. The CM kernel advances per token by (num_kv_heads * head_size) elements
//     (the "unpacked" pitch).  The packed buffer's actual token stride is 3*H*S,
//     so every K/V read from token 1 onwards lands on a wrong address.
//
// Topology mirroring the Qwen-VL Vision Merger after TransposeSplitMatcher
// (the VL_SDPA line; crop axis=1). NOTE: this packed-pitch bug lives in the
// vl_sdpa CM kernel only, so it is specific to the TransposeSplitMatcher ->
// VL_SDPA path, not the QKVSplitReshapeMatcher -> SDPA/RMSNorm path.
//   qkv [L,3,H,S]
//     -> crop(axis=1, slice=0) -> squeeze -> eltwise(*1) ->  q [L,H,S]  --
//     -> crop(axis=1, slice=1) -> squeeze -> eltwise(*1) ->  k [L,H,S]  -> vl_sdpa
//     -> crop(axis=1, slice=2) -> squeeze               ->  v [L,H,S]  /
//
// Expected: FAIL on the current branch (without the fix), PASS after fix.
// ============================================================================
TEST(vlsdpa_gpu_test, packed_qkv_inplace_crop_pitch_regression) {
    // ===========================================================================
    // Regression test for the vl_sdpa CM kernel packed-QKV pitch/offset handling.
    //
    // After TransposeSplitMatcher (Qwen-VL Vision Merger) fires:
    //   PackedQKV[-1,3,H,S] -> Split(axis=1) -> crop -> reshape -> Q/K/V -> VLSDPA
    // the three in-place crops all alias ONE packed [L, 3*H, S] buffer.  Each slice
    // therefore keeps the packed per-token stride 3*H*S and carries a feature-axis
    // padding that encodes its slice index:
    //   Q: lower_pad[f]=0,   upper_pad[f]=2H  (base offset 0)
    //   K: lower_pad[f]=H,   upper_pad[f]=H   (base offset H*S)
    //   V: lower_pad[f]=2H,  upper_pad[f]=0   (base offset 2H*S)
    //
    // The CM kernel must (a) use the packed stride 3*H*S per token
    // (CMFLA_IS_QKV_FUSED) and (b) offset each slice by lower_pad[f]*head_size.
    // The unfixed branch instead used the contiguous stride H*S and computed
    // token_offset = lower_pad[f]*num_q_heads*head_size (double-counting H), so it
    // read Q/K/V from the wrong addresses.  This test feeds the packed layout to
    // vl_sdpa and compares against a contiguous reference; it fails on the unfixed
    // branch and passes once the in-place-crop path is handled correctly.
    //
    // S=72 matches the real Omni/Qwen-VL head_size where the bug was observed.
    // ===========================================================================
    auto& engine = get_test_engine();
    ExecutionConfig check_cfg = get_test_default_config(engine);
    if (!engine.get_device_info().supports_immad ||
        !cldnn::check_cm_jit_support(engine, check_cfg))
        GTEST_SKIP() << "vl_sdpa CM kernel not available on this device";

    tests::random_generator rg(GET_SUITE_NAME);

    const int32_t H = 2, S = 72, L = 16;
    const std::vector<int32_t> cu_seqlens_vec = {0, L};

    // Explicit non-zero data so attention output is predictable and diverges
    // measurably when the kernel reads wrong Q/K/V positions.
    std::vector<ov::float16> q_data(L * H * S), k_data(L * H * S), v_data(L * H * S);
    for (int l = 0; l < L; l++) {
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                int idx = l * H * S + h * S + s;
                q_data[idx] = ov::float16(0.5f);
                k_data[idx] = ov::float16(static_cast<float>(l + 1) * 0.05f + s * 0.001f);
                v_data[idx] = ov::float16(static_cast<float>(l + 1) * 0.1f);
            }
        }
    }
    std::vector<ov::float16> mask_data(L * L, ov::float16(0.f));

    // -----------------------------------------------------------------------
    // Contiguous memories for the reference path.
    // -----------------------------------------------------------------------
    layout qhs_layout{{L, H, S}, data_types::f16, format::bfyx};
    auto q_mem    = engine.allocate_memory(qhs_layout);
    auto k_mem    = engine.allocate_memory(qhs_layout);
    auto v_mem    = engine.allocate_memory(qhs_layout);
    auto attn_mem = engine.allocate_memory({{1, L, L}, data_types::f16, format::bfyx});
    auto cu_mem   = engine.allocate_memory(
        {{static_cast<ov::Dimension::value_type>(cu_seqlens_vec.size())},
         data_types::i32, format::bfyx});

    set_values(q_mem,    q_data);
    set_values(k_mem,    k_data);
    set_values(v_mem,    v_data);
    set_values(attn_mem, mask_data);
    set_values(cu_mem,   cu_seqlens_vec);

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::optimize_data(true));

    const std::vector<int64_t>  ord_in  = {1, 0, 2};
    const std::vector<uint16_t> ord_out = {1, 0, 2};

    // -----------------------------------------------------------------------
    // Reference topology: contiguous Q/K/V -> SDPA (non-CM path).
    // -----------------------------------------------------------------------
    layout qhs_dyn{ov::PartialShape{-1, H, S}, data_types::f16, format::bfyx};
    layout attn_dyn{{1, -1, -1}, data_types::f16, format::bfyx};

    topology ref_topo;
    ref_topo.add(
        input_layout("q",    qhs_dyn),
        input_layout("k",    qhs_dyn),
        input_layout("v",    qhs_dyn),
        input_layout("attn", attn_dyn));
    ref_topo.add(scaled_dot_product_attention("sdpa",
        {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
        false, -1, ord_in, ord_in, ord_in, {0, 1, 2}, {}, false));
    ref_topo.add(permute("perm_o", input_info("sdpa"), ord_out));
    ref_topo.add(reorder("result", input_info("perm_o"), format::bfyx, data_types::f16));

    network ref_net(engine, ref_topo, cfg);
    ref_net.set_input_data("q",    q_mem);
    ref_net.set_input_data("k",    k_mem);
    ref_net.set_input_data("v",    v_mem);
    ref_net.set_input_data("attn", attn_mem);
    auto ref_out = ref_net.execute().at("result").get_memory();

    // -----------------------------------------------------------------------
    // Packed topology: each of Q/K/V is a padded [L,H,S] view (per-token stride
    // 3*H*S) into its own packed buffer, mirroring the three in-place crops that
    // Split(axis=1) produces from a packed [L,3,H,S] QKV tensor.
    //   Q: lower_pad[f]=0,   upper_pad[f]=2H
    //   K: lower_pad[f]=H,   upper_pad[f]=H
    //   V: lower_pad[f]=2H,  upper_pad[f]=0
    // -----------------------------------------------------------------------
    const auto lp_q = static_cast<tensor::value_type>(0);
    const auto up_q = static_cast<tensor::value_type>(2 * H);
    const auto lp_k = static_cast<tensor::value_type>(H);
    const auto up_k = static_cast<tensor::value_type>(H);
    const auto lp_v = static_cast<tensor::value_type>(2 * H);
    const auto up_v = static_cast<tensor::value_type>(0);

    auto make_packed = [&](tensor::value_type lower, tensor::value_type upper,
                           const std::vector<ov::float16>& src) {
        layout pad_layout{{L, H, S}, data_types::f16, format::bfyx,
                          cldnn::padding({0, lower, 0, 0}, {0, upper, 0, 0})};
        auto mem = engine.allocate_memory(pad_layout);
        cldnn::mem_lock<ov::float16, mem_lock_type::write> raw(mem, get_test_stream());
        const size_t phys_token_stride = static_cast<size_t>(3) * H * S;
        for (int l = 0; l < L; l++)
            for (int h = 0; h < H; h++)
                for (int s = 0; s < S; s++)
                    raw[static_cast<size_t>(l) * phys_token_stride
                        + static_cast<size_t>(lower + h) * S + s]
                        = src[static_cast<size_t>(l) * H * S + h * S + s];
        return mem;
    };

    auto q_mem_packed = make_packed(lp_q, up_q, q_data);
    auto k_mem_packed = make_packed(lp_k, up_k, k_data);
    auto v_mem_packed = make_packed(lp_v, up_v, v_data);

    layout cu_dyn{{-1}, data_types::i32, format::bfyx};
    auto pad_dyn = [&](tensor::value_type lower, tensor::value_type upper) {
        return layout{ov::PartialShape{-1, H, S}, data_types::f16, format::bfyx,
                      cldnn::padding({0, lower, 0, 0}, {0, upper, 0, 0})};
    };

    topology packed_topo;
    packed_topo.add(
        input_layout("q_in",       pad_dyn(lp_q, up_q)),
        input_layout("k_in",       pad_dyn(lp_k, up_k)),
        input_layout("v_in",       pad_dyn(lp_v, up_v)),
        input_layout("cu_seqlens", cu_dyn),
        vl_sdpa("vlsdpa",
                {input_info("q_in"), input_info("k_in"),
                 input_info("v_in"), input_info("cu_seqlens")},
                ord_in, ord_in, ord_in, ord_in),
        reorder("result", input_info("vlsdpa"), format::bfyx, data_types::f16));

    network packed_net(engine, packed_topo, cfg);
    packed_net.set_input_data("q_in",       q_mem_packed);
    packed_net.set_input_data("k_in",       k_mem_packed);
    packed_net.set_input_data("v_in",       v_mem_packed);
    packed_net.set_input_data("cu_seqlens", cu_mem);
    auto packed_out = packed_net.execute().at("result").get_memory();

    // Sanity: the packed padding must reach vl_sdpa on all three inputs, otherwise
    // the kernel would see contiguous buffers and the test would be meaningless.
    auto q_pad = packed_net.get_primitive("q_in")->get_output_layout().data_padding;
    auto k_pad = packed_net.get_primitive("k_in")->get_output_layout().data_padding;
    auto v_pad = packed_net.get_primitive("v_in")->get_output_layout().data_padding;
    ASSERT_EQ(k_pad._lower_size[1], lp_k) << "K lower_pad[f] must equal H=" << H;
    ASSERT_EQ(v_pad._lower_size[1], lp_v) << "V lower_pad[f] must equal 2H=" << (2 * H);
    ASSERT_EQ(q_pad._upper_size[1], up_q) << "Q upper_pad[f] must equal 2H=" << (2 * H);

    // Element-wise comparison: with the in-place-crop path handled correctly the
    // packed output matches the contiguous reference.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_out, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> pkd_ptr(packed_out, get_test_stream());
    ASSERT_EQ(ref_ptr.size(), pkd_ptr.size());

    // Guard against all-zero reference output (degenerate data → false pass).
    float ref_sum = 0.f;
    for (size_t i = 0; i < ref_ptr.size(); i++)
        ref_sum += std::abs(static_cast<float>(ref_ptr[i]));
    ASSERT_GT(ref_sum, 1.0f)
        << "Reference SDPA output is near-zero — test data degenerate, "
           "mismatches=0 comparison would be meaningless.";

    int64_t mismatches = 0;
    for (size_t i = 0; i < ref_ptr.size(); i++) {
        if (std::abs(static_cast<float>(ref_ptr[i]) - static_cast<float>(pkd_ptr[i])) > 0.05f)
            ++mismatches;
    }
    EXPECT_EQ(mismatches, 0)
        << "vl_sdpa produced " << mismatches << " wrong elements (out of "
        << ref_ptr.size() << ") for packed QKV (in-place crop) with S=" << S
        << ", H=" << H << ", L=" << L << ".\n"
        "Expected packed handling: per-token stride = 3*H*S and\n"
        "  token_offset_{q,k,v} = lower_pad[f]*head_size = {0, " << (H * S)
        << ", " << (2 * H * S) << "}.\n"
        "Unfixed branch used contiguous stride H*S and\n"
        "  token_offset = lower_pad[f]*num_q_heads*head_size (double-counts H).\n"
        "Fix: see openvino.mx PR#264.";
}

TEST(vlsdpa_gpu_test, contiguous_qkv_no_inplace_crop) {
    // =========================================================================
    // Negative / regression guard for the packed-QKV fix (openvino.mx PR#264).
    //
    // When Q/K/V arrive as independent contiguous [L,H,S] buffers (no in-place
    // crop, no feature-axis padding), the fix must NOT activate the packed path:
    //   CMFLA_IS_QKV_FUSED must be 0  →  per-token stride stays H*S (correct)
    //   token_offset_q/k/v must be 0  →  no base shift applied
    //
    // Any accidental mis-detection of "is_qkv_fused" here would break the normal
    // (non-packed) inference path.  This test catches that regression.
    // =========================================================================
    auto& engine = get_test_engine();
    ExecutionConfig check_cfg = get_test_default_config(engine);
    if (!engine.get_device_info().supports_immad ||
        !cldnn::check_cm_jit_support(engine, check_cfg))
        GTEST_SKIP() << "vl_sdpa CM kernel not available on this device";

    const int32_t H = 2, S = 72, L = 16;
    const std::vector<int32_t> cu_seqlens_vec = {0, L};

    // Same data as the packed test for easy cross-comparison.
    std::vector<ov::float16> q_data(L * H * S), k_data(L * H * S), v_data(L * H * S);
    for (int l = 0; l < L; l++) {
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                int idx = l * H * S + h * S + s;
                q_data[idx] = ov::float16(0.5f);
                k_data[idx] = ov::float16(static_cast<float>(l + 1) * 0.05f + s * 0.001f);
                v_data[idx] = ov::float16(static_cast<float>(l + 1) * 0.1f);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fully contiguous memories — no padding whatsoever.
    // -----------------------------------------------------------------------
    layout qhs_layout{{L, H, S}, data_types::f16, format::bfyx};
    auto q_mem    = engine.allocate_memory(qhs_layout);
    auto k_mem    = engine.allocate_memory(qhs_layout);
    auto v_mem    = engine.allocate_memory(qhs_layout);
    auto cu_mem   = engine.allocate_memory(
        {{static_cast<ov::Dimension::value_type>(cu_seqlens_vec.size())},
         data_types::i32, format::bfyx});

    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(cu_mem, cu_seqlens_vec);

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::optimize_data(true));

    const std::vector<int64_t>  ord_in  = {1, 0, 2};
    const std::vector<uint16_t> ord_out = {1, 0, 2};

    // -----------------------------------------------------------------------
    // Reference: contiguous Q/K/V → SDPA (non-CM path).
    // -----------------------------------------------------------------------
    layout qhs_dyn{ov::PartialShape{-1, H, S}, data_types::f16, format::bfyx};
    layout attn_dyn{{1, -1, -1}, data_types::f16, format::bfyx};

    std::vector<ov::float16> mask_data(L * L, ov::float16(0.f));
    auto attn_mem = engine.allocate_memory({{1, L, L}, data_types::f16, format::bfyx});
    set_values(attn_mem, mask_data);

    topology ref_topo;
    ref_topo.add(
        input_layout("q",    qhs_dyn),
        input_layout("k",    qhs_dyn),
        input_layout("v",    qhs_dyn),
        input_layout("attn", attn_dyn));
    ref_topo.add(scaled_dot_product_attention("sdpa",
        {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
        false, -1, ord_in, ord_in, ord_in, {0, 1, 2}, {}, false));
    ref_topo.add(permute("perm_o", input_info("sdpa"), ord_out));
    ref_topo.add(reorder("result", input_info("perm_o"), format::bfyx, data_types::f16));

    network ref_net(engine, ref_topo, cfg);
    ref_net.set_input_data("q",    q_mem);
    ref_net.set_input_data("k",    k_mem);
    ref_net.set_input_data("v",    v_mem);
    ref_net.set_input_data("attn", attn_mem);
    auto ref_out = ref_net.execute().at("result").get_memory();

    // -----------------------------------------------------------------------
    // VL_SDPA with contiguous (no-padding) Q/K/V.
    // -----------------------------------------------------------------------
    layout cu_dyn{{-1}, data_types::i32, format::bfyx};

    topology vlsdpa_topo;
    vlsdpa_topo.add(
        input_layout("q_in",       qhs_dyn),
        input_layout("k_in",       qhs_dyn),
        input_layout("v_in",       qhs_dyn),
        input_layout("cu_seqlens", cu_dyn),
        vl_sdpa("vlsdpa",
                {input_info("q_in"), input_info("k_in"),
                 input_info("v_in"), input_info("cu_seqlens")},
                ord_in, ord_in, ord_in, ord_in),
        reorder("result", input_info("vlsdpa"), format::bfyx, data_types::f16));

    network vlsdpa_net(engine, vlsdpa_topo, cfg);
    vlsdpa_net.set_input_data("q_in",       q_mem);
    vlsdpa_net.set_input_data("k_in",       k_mem);
    vlsdpa_net.set_input_data("v_in",       v_mem);
    vlsdpa_net.set_input_data("cu_seqlens", cu_mem);
    auto out = vlsdpa_net.execute().at("result").get_memory();

    // Sanity: all paddings must be zero — CMFLA_IS_QKV_FUSED must NOT fire.
    auto q_pad = vlsdpa_net.get_primitive("q_in")->get_output_layout().data_padding;
    auto k_pad = vlsdpa_net.get_primitive("k_in")->get_output_layout().data_padding;
    auto v_pad = vlsdpa_net.get_primitive("v_in")->get_output_layout().data_padding;
    ASSERT_EQ(q_pad._lower_size[1], 0) << "Q must have no feature padding (contiguous path)";
    ASSERT_EQ(k_pad._lower_size[1], 0) << "K must have no feature padding (contiguous path)";
    ASSERT_EQ(v_pad._lower_size[1], 0) << "V must have no feature padding (contiguous path)";
    ASSERT_EQ(q_pad._upper_size[1], 0) << "Q must have no upper feature padding";
    ASSERT_EQ(k_pad._upper_size[1], 0) << "K must have no upper feature padding";
    ASSERT_EQ(v_pad._upper_size[1], 0) << "V must have no upper feature padding";

    // Element-wise comparison: contiguous vl_sdpa must match the SDPA reference.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_out, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(out, get_test_stream());
    ASSERT_EQ(ref_ptr.size(), out_ptr.size());

    // Guard against the "all-zero output → false pass" trap: if the reference
    // is all-zero, mismatches=0 is trivially true even when the kernel is wrong
    // (the same issue seen with random near-zero f16 data in earlier debugging).
    // With our explicit data (q=0.5, k/v per-token ramps) the SDPA output must
    // be non-zero and non-trivial; lock that pre-condition in place.
    float ref_sum = 0.f;
    for (size_t i = 0; i < ref_ptr.size(); i++)
        ref_sum += std::abs(static_cast<float>(ref_ptr[i]));
    ASSERT_GT(ref_sum, 1.0f)
        << "Reference SDPA output is near-zero — test data degenerate, "
           "mismatches=0 comparison would be meaningless.";

    int64_t mismatches = 0;
    for (size_t i = 0; i < ref_ptr.size(); i++) {
        if (std::abs(static_cast<float>(ref_ptr[i]) - static_cast<float>(out_ptr[i])) > 0.05f)
            ++mismatches;
    }
    EXPECT_EQ(mismatches, 0)
        << "vl_sdpa produced " << mismatches << " wrong elements (out of "
        << ref_ptr.size() << ") on contiguous (non-packed) Q/K/V.\n"
        "CMFLA_IS_QKV_FUSED must be 0 and all token_offset_* must be 0 "
        "when no in-place crop is active (no feature-axis padding).\n"
        "S=" << S << ", H=" << H << ", L=" << L << ".";
}
