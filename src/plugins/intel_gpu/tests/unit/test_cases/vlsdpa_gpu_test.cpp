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

struct mixed_pitch_case_params {
    int32_t num_heads;
    int32_t head_size;
    std::vector<int32_t> cu_seqlens;
    int32_t q_lower_pad_heads;
    int32_t k_lower_pad_heads;
    int32_t v_lower_pad_heads;
};

static void run_mixed_pitch_case(const mixed_pitch_case_params& p,
                                 const bool strict_elementwise = false) {
    if (!vlsdpa_gpu_test::check_vlsdpa_available())
        GTEST_SKIP();

    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    rg.set_seed(GET_SUITE_NAME);

    const int32_t H = p.num_heads;
    const int32_t S = p.head_size;
    const int32_t L = p.cu_seqlens.back();

    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_o = {1, 0, 2};
    const std::vector<uint16_t> order_o2 = {1, 0, 2};

    layout q_dyn({-1, H, S}, data_types::f16, format::bfyx);
    layout k_dyn({-1, H, S}, data_types::f16, format::bfyx);
    layout v_dyn({-1, H, S}, data_types::f16, format::bfyx);
    layout attn_mask_dyn({1, -1, -1}, data_types::f16, format::bfyx);
    layout cu_dyn({-1}, data_types::i32, format::bfyx);

    layout q_static({L, H, S}, data_types::f16, format::bfyx);
    layout k_static({L, H, S}, data_types::f16, format::bfyx);
    layout v_static({L, H, S}, data_types::f16, format::bfyx);
    layout attn_mask_static({1, L, L}, data_types::f16, format::bfyx);
    layout cu_static({static_cast<ov::Dimension::value_type>(p.cu_seqlens.size())}, data_types::i32, format::bfyx);

    // Per-input lower feature padding emulates independent runtime base offsets / token pitches.
    layout q_padded_static({L, H, S}, data_types::f16, format::bfyx,
                           padding({0, p.q_lower_pad_heads, 0, 0}, {0, 0, 0, 0}));
    layout k_padded_static({L, H, S}, data_types::f16, format::bfyx,
                           padding({0, p.k_lower_pad_heads, 0, 0}, {0, 0, 0, 0}));
    layout v_padded_static({L, H, S}, data_types::f16, format::bfyx,
                           padding({0, p.v_lower_pad_heads, 0, 0}, {0, 0, 0, 0}));

    topology ref_topo;
    ref_topo.add(input_layout("q", q_dyn));
    ref_topo.add(input_layout("k", k_dyn));
    ref_topo.add(input_layout("v", v_dyn));
    ref_topo.add(input_layout("attn", attn_mask_dyn));
    ref_topo.add(scaled_dot_product_attention("sdpa",
                    {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
                    false, -1, order_q, order_k, order_v, {0, 1, 2}, {}, false));
    ref_topo.add(permute("permute_o", input_info("sdpa"), order_o2));
    ref_topo.add(reorder("result", input_info("permute_o"), format::bfyx, data_types::f16));

    topology opt_topo;
    opt_topo.add(input_layout("q_in", q_dyn));
    opt_topo.add(input_layout("k_in", k_dyn));
    opt_topo.add(input_layout("v_in", v_dyn));
    opt_topo.add(input_layout("attn", cu_dyn));

    const std::string q_name = p.q_lower_pad_heads > 0 ? "q_padded" : "q_in";
    const std::string k_name = p.k_lower_pad_heads > 0 ? "k_padded" : "k_in";
    const std::string v_name = p.v_lower_pad_heads > 0 ? "v_padded" : "v_in";

    if (p.q_lower_pad_heads > 0) {
        opt_topo.add(reorder("q_padded", input_info("q_in"), q_padded_static));
    }
    if (p.k_lower_pad_heads > 0) {
        opt_topo.add(reorder("k_padded", input_info("k_in"), k_padded_static));
    }
    if (p.v_lower_pad_heads > 0) {
        opt_topo.add(reorder("v_padded", input_info("v_in"), v_padded_static));
    }

    opt_topo.add(vl_sdpa("vlsdpa",
                    {input_info(q_name), input_info(k_name), input_info(v_name), input_info("attn")},
                    order_q, order_k, order_v, order_o));
    opt_topo.add(reorder("result", input_info("vlsdpa"), format::bfyx, data_types::f16));

    auto q_mem = engine.allocate_memory(q_static);
    auto k_mem = engine.allocate_memory(k_static);
    auto v_mem = engine.allocate_memory(v_static);
    auto attn_mask_mem = engine.allocate_memory(attn_mask_static);
    auto cu_mem = engine.allocate_memory(cu_static);

    auto q_data = rg.generate_random_1d<ov::float16>(L * H * S, -0.5f, 0.5f);
    auto k_data = rg.generate_random_1d<ov::float16>(L * H * S, -0.5f, 0.5f);
    auto v_data = rg.generate_random_1d<ov::float16>(L * H * S, -0.5f, 0.5f);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);

    std::vector<ov::float16> attn_mask_data(static_cast<size_t>(L) * static_cast<size_t>(L), -std::numeric_limits<ov::float16>::infinity());
    for (size_t i = 1; i < p.cu_seqlens.size(); ++i) {
        size_t start = static_cast<size_t>(p.cu_seqlens[i - 1]);
        size_t end = static_cast<size_t>(p.cu_seqlens[i]);
        for (size_t r = start; r < end; ++r) {
            for (size_t c = start; c < end; ++c) {
                attn_mask_data[r * static_cast<size_t>(L) + c] = ov::float16(0.0f);
            }
        }
    }
    set_values(attn_mask_mem, attn_mask_data);
    set_values(cu_mem, p.cu_seqlens);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto ref_net = get_network(engine, ref_topo, config, get_test_stream_ptr(), false);
    ref_net->set_input_data("q", q_mem);
    ref_net->set_input_data("k", k_mem);
    ref_net->set_input_data("v", v_mem);
    ref_net->set_input_data("attn", attn_mask_mem);
    auto ref_out = ref_net->execute().at("result").get_memory();

    auto opt_net = get_network(engine, opt_topo, config, get_test_stream_ptr(), false);
    opt_net->set_input_data("q_in", q_mem);
    opt_net->set_input_data("k_in", k_mem);
    opt_net->set_input_data("v_in", v_mem);
    opt_net->set_input_data("attn", cu_mem);
    auto opt_out = opt_net->execute().at("result").get_memory();

    const auto q_layout_runtime = opt_net->get_primitive(q_name)->get_output_layout();
    const auto k_layout_runtime = opt_net->get_primitive(k_name)->get_output_layout();
    const auto v_layout_runtime = opt_net->get_primitive(v_name)->get_output_layout();

    ASSERT_EQ(static_cast<int32_t>(q_layout_runtime.get_linear_offset()), p.q_lower_pad_heads * S);
    ASSERT_EQ(static_cast<int32_t>(k_layout_runtime.get_linear_offset()), p.k_lower_pad_heads * S);
    ASSERT_EQ(static_cast<int32_t>(v_layout_runtime.get_linear_offset()), p.v_lower_pad_heads * S);

    const int32_t expected_q_pitch = (p.q_lower_pad_heads + H) * S;
    const int32_t expected_k_pitch = (p.k_lower_pad_heads + H) * S;
    const int32_t expected_v_pitch = (p.v_lower_pad_heads + H) * S;
    ASSERT_EQ(static_cast<int32_t>(q_layout_runtime.get_pitches()[0]), expected_q_pitch);
    ASSERT_EQ(static_cast<int32_t>(k_layout_runtime.get_pitches()[0]), expected_k_pitch);
    ASSERT_EQ(static_cast<int32_t>(v_layout_runtime.get_pitches()[0]), expected_v_pitch);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(ref_out, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(opt_out, get_test_stream());

    ASSERT_EQ(ref_data.size(), opt_data.size())
        << "Mismatched output element count between reference and VLSDPA paths";

    const float ref_magnitude = std::sqrt(std::inner_product(ref_data.begin(), ref_data.end(), ref_data.begin(), 0.0f));
    const float opt_magnitude = std::sqrt(std::inner_product(opt_data.begin(), opt_data.end(), opt_data.begin(), 0.0f));
    ASSERT_GT(ref_magnitude, 0.0f) << "Reference output has zero magnitude";
    ASSERT_GT(opt_magnitude, 0.0f) << "VLSDPA output has zero magnitude";

    float max_abs_diff = 0.0f;
    for (size_t idx = 0; idx < ref_data.size(); ++idx) {
        const float ref_val = static_cast<float>(ref_data[idx]);
        const float opt_val = static_cast<float>(opt_data[idx]);
        ASSERT_FALSE(std::isnan(ref_val) || std::isnan(opt_val)) << "NaN found at index " << idx;
        max_abs_diff = std::max(max_abs_diff, std::abs(ref_val - opt_val));
    }

    ASSERT_GE(cosineSimilarity(ref_data, opt_data), 0.95f);
    if (strict_elementwise) {
        ASSERT_LE(max_abs_diff, 0.10f) << "Max abs diff is too large for strict mixed-pitch check";
    }
}

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

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_v_padded) {
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          0 /*q_lower_pad_heads*/, 0 /*k_lower_pad_heads*/, 4 /*v_lower_pad_heads*/});
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_control_contiguous) {
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          0 /*q_lower_pad_heads*/, 0 /*k_lower_pad_heads*/, 0 /*v_lower_pad_heads*/});
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_q_padded) {
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          4 /*q_lower_pad_heads*/, 0 /*k_lower_pad_heads*/, 0 /*v_lower_pad_heads*/});
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_k_padded) {
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          0 /*q_lower_pad_heads*/, 4 /*k_lower_pad_heads*/, 0 /*v_lower_pad_heads*/});
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_qkv_all_padded) {
    // Distinct lower pads per input enforce different base offsets for all three streams.
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          2 /*q_lower_pad_heads*/, 4 /*k_lower_pad_heads*/, 6 /*v_lower_pad_heads*/}, true);
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_qk_equal_v_diff_dynamic_l) {
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16} /*cu_seqlens*/,
                          0 /*q_lower_pad_heads*/, 0 /*k_lower_pad_heads*/, 4 /*v_lower_pad_heads*/});
    run_mixed_pitch_case({2 /*H*/, 72 /*S*/, {0, 16, 32} /*cu_seqlens*/,
                          0 /*q_lower_pad_heads*/, 0 /*k_lower_pad_heads*/, 4 /*v_lower_pad_heads*/});
}

TEST(vlsdpa_gpu_test, mixed_input_runtime_pitches_headsize_param) {
    const std::vector<mixed_pitch_case_params> cases = {
        {1, 64,  {0, 16}, 0, 0, 2},
        {1, 72,  {0, 16}, 0, 0, 2},
        {2, 64,  {0, 16}, 0, 0, 4},
        {2, 72,  {0, 16}, 0, 0, 4},
        {2, 128, {0, 16}, 0, 0, 4},
    };

    for (const auto& test_case : cases) {
        run_mixed_pitch_case(test_case);
    }
}

} // namespace
