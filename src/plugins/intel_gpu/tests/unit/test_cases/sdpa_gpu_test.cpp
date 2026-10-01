// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "dpas_backend_test_helper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "impls/ocl_v2/sdpa/sdpa_ocl_utils.hpp"
#include "impls/ocl_v2/sdpa/sdpa_opt.hpp"
#include "impls/ocl_v2/sdpa/sdpa_ref.hpp"
#include "openvino/reference/scaled_dot_product_attention.hpp"
#include "openvino/util/file_util.hpp"
#include "program_wrapper.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include "scaled_dot_product_attention_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;
// #define ENABLE_ONEDNN_FOR_GPU
namespace  {
#ifdef ENABLE_ONEDNN_FOR_GPU
struct sdpa_test_params {
    int head_size;
    int num_heads;
    int kv_num_heads;
    int sequence_length_q;
    int sequence_length_kv;
    int batch;
    bool dynamic;
    bool use_scalar_scale_val;
    float scale_val;
    bool use_scalar_attn_mask;
    float attn_mask_val;
    int bit_width;
    bool asymmetric;
    data_types dt;

    // Constructor for basic tests (backward compatibility)
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool dynamic_shape, data_types dt = data_types::f16)
          : head_size(h_size), num_heads(n_heads), kv_num_heads(n_heads), sequence_length_q(seq_q),
          sequence_length_kv(seq_kv), batch(b), dynamic(dynamic_shape),
          use_scalar_scale_val(false), scale_val(1.0f), use_scalar_attn_mask(false),
            attn_mask_val(0.0f), bit_width(0), asymmetric(false), dt(dt) {}

    // Constructor for advanced caching tests
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool use_scale, float scale, bool use_mask, float mask, data_types dt = data_types::f16)
          : head_size(h_size), num_heads(n_heads), kv_num_heads(n_heads), sequence_length_q(seq_q), sequence_length_kv(seq_kv),
          batch(b), dynamic(true), use_scalar_scale_val(use_scale),
            scale_val(scale), use_scalar_attn_mask(use_mask), attn_mask_val(mask), bit_width(0), asymmetric(false), dt(dt){}
    
    // constructor for quantization tests
    sdpa_test_params(int h_size, int q_heads, int kv_heads, int seq_q, int seq_kv, int b, int bits, bool asym, data_types dt = data_types::f16)
          : head_size(h_size), num_heads(q_heads), kv_num_heads(kv_heads), sequence_length_q(seq_q),
            sequence_length_kv(seq_kv), batch(b), dynamic(false), use_scalar_scale_val(false), scale_val(1.0f),
            use_scalar_attn_mask(false), attn_mask_val(0.0f), bit_width(bits), asymmetric(asym), dt(dt) {}
};

struct sdpa_gpu_test : public ::testing::TestWithParam<sdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx, data_types dt = data_types::f16) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        if (dt == data_types::bf16) {
            auto input_data = rg.generate_random_1d<ov::bfloat16>(size, -1.0f, 1.0f);
            set_values(mem, input_data);
        } else {
            auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
            set_values(mem, input_data);
        }
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(bool is_caching_test, bool use_optimized_sdpa,
            cldnn::layout input0_layout,
            cldnn::layout input1_layout,
            cldnn::layout input2_layout,
            cldnn::layout input3_layout,
            cldnn::memory::ptr input0,
            cldnn::memory::ptr input1,
            cldnn::memory::ptr input2,
            cldnn::memory::ptr input3,
            bool use_scalar_scale_val = false,
            float scale_val = 1.0f,
            bool use_scalar_attn_mask = false,
            float attn_mask_val = 0.0f,
            data_types dt = data_types::f16) {
        auto& engine = get_test_engine();
        topology topo;
        topo.add(input_layout("input0", input0_layout));
        topo.add(input_layout("input1", input1_layout));
        topo.add(input_layout("input2", input2_layout));
        topo.add(input_layout("input3", input3_layout));

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
            false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false);

        if (use_scalar_scale_val) {
            sdpa_prim.scale_val = scale_val;
        }

        if (use_scalar_attn_mask) {
            sdpa_prim.attn_mask_val = attn_mask_val;
        }

        topo.add(sdpa_prim);
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, dt));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_optimized_sdpa) {
            if (!is_caching_test) {
                if (engine.get_device_info().supports_immad) {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
                } else {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_opt"}} }));
                }
            }
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                   {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);
        net->set_input_data("input3", input3);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(sdpa_test_params& p, bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto seq_length_q = p.sequence_length_q;
        const auto seq_length_kv = p.sequence_length_kv;
        const auto batch = p.batch;
        const auto use_scalar_scale_val = p.use_scalar_scale_val;
        const auto scale_val = p.scale_val;
        const auto use_scalar_attn_mask = p.use_scalar_attn_mask;
        const auto attn_mask_val = p.attn_mask_val;
        const auto test_two_rank_mask = p.sequence_length_q == p.sequence_length_kv ? true : false;

        auto& engine = get_test_engine();
        cldnn::layout input0_layout, input1_layout, input2_layout, input3_layout;
        cldnn::layout input0_static_layout, input1_static_layout, input2_static_layout, input3_static_layout;

        if (p.dynamic) {
            input0_layout = cldnn::layout({-1, -1, num_heads, head_size}, p.dt, format::bfyx);
            input1_layout = cldnn::layout({-1, -1, num_heads, head_size}, p.dt, format::bfyx);
            input2_layout = cldnn::layout({-1, -1, num_heads, head_size}, p.dt, format::bfyx);

            if (test_two_rank_mask) {
                input3_layout = cldnn::layout({ -1, -1}, p.dt, format::bfyx);
            } else {
                input3_layout = cldnn::layout({-1, num_heads, -1, -1}, p.dt, format::bfyx);
            }

            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, p.dt, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, p.dt, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, p.dt, format::bfyx);
            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, p.dt, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, p.dt, format::bfyx);
            }
        } else {
            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, p.dt, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, p.dt, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, p.dt, format::bfyx);

            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, p.dt, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, p.dt, format::bfyx);
            }

            input0_layout = input0_static_layout;
            input1_layout = input1_static_layout;
            input2_layout = input2_static_layout;
            input3_layout = input3_static_layout;
        }

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);
        auto input3 = engine.allocate_memory(input3_static_layout);

        load_input(input0, 0, p.dt);
        load_input(input1, 1, p.dt);
        load_input(input2, 2, p.dt);
        load_input(input3, 3, input3_static_layout.data_type);

        auto [mem_ref_ptr, net_ref_ptr] = run_network(is_caching_test, false,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val, p.dt);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(is_caching_test, true,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val, p.dt);

        if (is_caching_test) {
            auto inst = net_opt_ptr->get_primitive("sdpa");
            auto& sdpa_node = inst->get_node().as<scaled_dot_product_attention>();

            if (use_scalar_scale_val) {
                ASSERT_TRUE(sdpa_node.get_primitive()->scale_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->scale_val.value(), scale_val);
            }

            if (use_scalar_attn_mask) {
                ASSERT_TRUE(sdpa_node.get_primitive()->attn_mask_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->attn_mask_val.value(), attn_mask_val);
            }
        }

        if (p.dt == data_types::bf16) {
            cldnn::mem_lock<ov::bfloat16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
            cldnn::mem_lock<ov::bfloat16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(static_cast<float>(opt_data[idx])) || std::isnan(static_cast<float>(ref_data[idx]))) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        } else {
            cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
            cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        std::string result = "sdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               std::to_string(info.param.sequence_length_q) + "_" +
               std::to_string(info.param.sequence_length_kv) + "_" +
               std::to_string(info.param.batch);

        if (info.param.use_scalar_scale_val) {
            result += "_scale_" + std::to_string(static_cast<int>(info.param.scale_val * 1000));
        }

        if (info.param.use_scalar_attn_mask) {
            result += "_mask_" + std::to_string(static_cast<int>(info.param.attn_mask_val * 1000));
        }

        if (!info.param.dynamic) {
            result += "_static";
        }

        if (info.param.dt == data_types::bf16) {
            result += "_bf16";
        }

        return result;
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_gpu_test,
    sdpa_gpu_test,
    ::testing::Values(
        sdpa_test_params{64, 32, 990, 128, 2, true}, // dynamic
        sdpa_test_params{64, 32, 990, 128, 2, false}, // static
        sdpa_test_params{64, 32, 990, 1, 2, true}, // dynamic
        sdpa_test_params{64, 32, 990, 1, 2, false}, // static
        sdpa_test_params{64, 10, 77, 77, 1, true}, // two ranks mask
        sdpa_test_params{64, 10, 77, 77, 1, false}, // two ranks mask
        sdpa_test_params{64, 32, 128, 128, 2, true, 0.125f, false, 0.0f},  // scale_val only
        sdpa_test_params{64, 32, 128, 128, 2, false, 1.0f, true, 0.5f},     // attn_mask only
        sdpa_test_params{512, 8, 1, 1024, 2, true},
        sdpa_test_params{64, 32, 128, 128, 2, true, data_types::bf16},   // bf16 dynamic
        sdpa_test_params{64, 32, 128, 128, 2, false, data_types::bf16},  // bf16 static
        sdpa_test_params{64, 10, 77, 77, 1, true, data_types::bf16}      // bf16 two ranks mask
    ),
    sdpa_gpu_test::PrintToStringParamName
);

TEST_P(sdpa_gpu_test, basic) {
    auto p = GetParam();
    execute(p);
}

TEST_P(sdpa_gpu_test, basic_caching) {
    auto p = GetParam();
    execute(p, true);
}

// Test that an explicit causal attention mask produces the same result as is_causal=true.
static void run_sdpa_causal_mask(int batch, int q_num_heads, int kv_num_heads,
                                 int seq_q, int seq_kv, int head_size, bool causal_lower_right = true) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto q_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * q_num_heads * seq_q * head_size, -1.0f, 1.0f);
    auto k_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size, -1.0f, 1.0f);
    auto v_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size, -1.0f, 1.0f);

    // Build causal attention mask with the requested alignment: shape [1, 1, seq_q, seq_kv]
    // 0 for valid positions (row >= col offset), -inf for masked positions.
    const size_t mask_size = static_cast<size_t>(seq_q) * seq_kv;
    std::vector<ov::float16> mask_data(mask_size);
    const int col_offset = causal_lower_right ? seq_kv - seq_q : 0;
    for (int r = 0; r < seq_q; ++r) {
        for (int c = 0; c < seq_kv; ++c) {
            if (c <= r + col_offset) {
                mask_data[r * seq_kv + c] = ov::float16(0.0f);
            } else {
                mask_data[r * seq_kv + c] = ov::float16(-INFINITY);
            }
        }
    }

    const layout q_layout({batch, q_num_heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout kv_layout({batch, kv_num_heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout mask_layout({1, 1, seq_q, seq_kv}, data_types::f16, format::bfyx);

    const layout q_dyn_layout({batch, q_num_heads, -1, head_size}, data_types::f16, format::bfyx);
    const layout kv_dyn_layout({batch, kv_num_heads, -1, head_size}, data_types::f16, format::bfyx);
    const layout mask_dyn_layout({1, 1, -1, -1}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(kv_layout);
    auto v_mem = engine.allocate_memory(kv_layout);
    auto mask_mem = engine.allocate_memory(mask_layout);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(mask_mem, mask_data);

    // --- Golden reference: is_causal=false, explicit mask as 4th input, static shapes ---
    auto make_ref_output = [&]() {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_layout));
        topo.add(input_layout("v", kv_layout));
        topo.add(input_layout("mask", mask_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v"), input_info("mask")},
                                                 false, -1,
                                                 {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3},
                                                 {}, false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("mask", mask_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Optimized path: is_causal=true, no mask input, dynamic shapes ---
    auto make_opt_output = [&]() {
        topology topo;
        topo.add(input_layout("q", q_dyn_layout));
        topo.add(input_layout("k", kv_dyn_layout));
        topo.add(input_layout("v", kv_dyn_layout));
        auto prim = causal_lower_right
                        ? scaled_dot_product_attention("sdpa",
                                                       {input_info("q"), input_info("k"), input_info("v")},
                                                       true,
                                                       -1,
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {},
                                                       false,
                                                       true)
                        : scaled_dot_product_attention("sdpa",
                                                       {input_info("q"), input_info("k"), input_info("v")},
                                                       true,
                                                       -1,
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {0, 1, 2, 3},
                                                       {},
                                                       false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < ref_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(ref_ptr[i]))) << "NaN in explicit mask output at index " << i;
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in is_causal output at index " << i;
    }

    const float sim = cosineSimilarity(ref_ptr, opt_ptr);
    ASSERT_GE(sim, 0.99f) << "explicit mask vs is_causal cosine similarity too low: " << sim;
}

TEST(sdpa_gpu_causal_mask, prefill_40q_40kv_512seq) {
    run_sdpa_causal_mask(1, 40, 40, 512, 512, 128);
}

TEST(sdpa_gpu_causal_mask, prefill_upper_left_default_40q_40kv_512seq) {
    run_sdpa_causal_mask(1, 40, 40, 512, 512, 128, false);
}

TEST(sdpa_gpu_causal_mask, decode_40q_40kv_512seq) {
    run_sdpa_causal_mask(1, 40, 40, 1, 512, 128);
}

TEST(sdpa_gpu_causal_mask, prefill_40q_10kv_512seq) {
    run_sdpa_causal_mask(1, 40, 10, 512, 512, 128);
}

TEST(sdpa_gpu_causal_mask, decode_40q_10kv_444seq) {
    run_sdpa_causal_mask(1, 40, 10, 1, 444, 128);
}

TEST(sdpa_gpu_causal_mask, decode_32q_8kv_1024seq) {
    run_sdpa_causal_mask(1, 32, 8, 1, 1024, 128);
}

struct micro_sdpa_prefetch_k_params {
    int head_size;
    int num_heads;
    int seq_len_q;
    int seq_len_kv;
    bool is_causal;
};

class sdpa_micro_prefetch_k_test : public ::testing::TestWithParam<micro_sdpa_prefetch_k_params> {
public:
    static std::string PrintToStringParamName(const testing::TestParamInfo<micro_sdpa_prefetch_k_params>& info) {
        const auto& p = info.param;
        return "d" + std::to_string(p.head_size) + "_h" + std::to_string(p.num_heads) + "_q" +
               std::to_string(p.seq_len_q) + "_kv" + std::to_string(p.seq_len_kv) +
               (p.is_causal ? "_causal" : "_full");
    }
};

TEST_P(sdpa_micro_prefetch_k_test, multi_tile_k_runs_micro_sdpa) {
    auto& engine = get_test_engine();
    const auto& device_info = engine.get_device_info();
    const auto p = GetParam();

    if (!device_info.supports_immad)
        GTEST_SKIP() << "sdpa_micro requires a device with systolic (immad) support";
    if (device_info.arch < cldnn::gpu_arch::xe_hpc)
        GTEST_SKIP() << "PREFETCH_K0/PREFETCH_K are only emitted for arch >= xe_hpc; this device "
                        "runs sdpa_micro without the prefetch under test"; 
    if (device_info.arch == cldnn::gpu_arch::xe3p && p.head_size <= 64)
        GTEST_SKIP() << "micro SDPA is disabled on xe3p for head_size <= 64";

    const ov::Shape q_shape{1, static_cast<size_t>(p.num_heads), static_cast<size_t>(p.seq_len_q),
                            static_cast<size_t>(p.head_size)};
    const ov::Shape kv_shape{1, static_cast<size_t>(p.num_heads), static_cast<size_t>(p.seq_len_kv),
                             static_cast<size_t>(p.head_size)};

    const layout q_layout(q_shape, data_types::f16, format::bfyx);
    const layout kv_layout(kv_shape, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(kv_layout);
    auto v_mem = engine.allocate_memory(kv_layout);

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto fill_random = [&](const memory::ptr& mem) {
        set_values(mem, rg.generate_random_1d<ov::float16>(mem->get_layout().count(), -1.0f, 1.0f));
    };
    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);

    // Fresh topology per run: dropping the redundant trailing reorder renames the SDPA node in
    // place, which mutates the topology object.
    auto make_topology = [&]() {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_layout));
        topo.add(input_layout("v", kv_layout));
        topo.add(scaled_dot_product_attention("sdpa",
                                              {input_info("q"), input_info("k"), input_info("v")},
                                              p.is_causal,
                                              -1,
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {},
                                              false));
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));
        return topo;
    };

    auto run_network = [&]() {
        auto topology = make_topology();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        // Advisory only: kernel_name is read by the legacy impls/ocl selector, not by ocl_v2.
        // The assertion below is what guarantees micro SDPA ran.
        config.set_property(ov::intel_gpu::force_implementations(
            ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_micro"}}}));

        auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        auto output = network->execute().at("result").get_memory();
        return std::make_pair(network, output);
    };

    // Look up by type, not id: the node is renamed to "result" when the reorder is dropped. The
    // node description carries the selected OpenCL entry point; kernel_id only has the impl class.
    auto selected_sdpa_kernel = [](const cldnn::network::ptr& net) {
        for (const auto& info : net->get_primitives_info()) {
            if (info.type_id == "scaled_dot_product_attention")
                return net->get_primitive_info(info.original_id);
        }
        return std::string{};
    };

    // A GPU fault from the prefetch would surface here as CL_OUT_OF_RESOURCES out of execute().
    auto [network, output] = run_network();
    const auto sdpa_info = selected_sdpa_kernel(network);
    ASSERT_FALSE(sdpa_info.empty()) << "no scaled_dot_product_attention node in the built program";
    ASSERT_TRUE(sdpa_info.find("sdpa_micro") != std::string::npos || sdpa_info.find("sdpa_ocl") != std::string::npos)
        << "Neither sdpa_micro nor sdpa_ocl was selected; the multi-K-tile prefetch path was not exercised. Node "
           "description was:\n"
        << sdpa_info;

    // No non-micro reference exists to compare against, and a prefetch cannot change results
    // anyway; this is coverage. Verified: the pre-fix ordering also passes here.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_data(output, get_test_stream());
    ASSERT_EQ(output_data.size(), ov::shape_size(q_shape));
    bool all_zero = true;
    for (size_t i = 0; i < output_data.size(); ++i) {
        const float v = static_cast<float>(output_data[i]);
        ASSERT_TRUE(std::isfinite(v)) << "non-finite output at index " << i;
        all_zero = all_zero && (v == 0.0f);
    }
    ASSERT_FALSE(all_zero) << "output is entirely zero";
}

INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_micro_prefetch_k,
    sdpa_micro_prefetch_k_test,
    ::testing::Values(
        micro_sdpa_prefetch_k_params{64, 4, 300, 1000, false},
        micro_sdpa_prefetch_k_params{64, 4, 300, 1000, true},
        micro_sdpa_prefetch_k_params{128, 2, 300, 1000, true},
        micro_sdpa_prefetch_k_params{256, 2, 177, 177, true}
    ),
    sdpa_micro_prefetch_k_test::PrintToStringParamName
);


TEST(sdpa_gpu_micro, transposed_v_matches_non_transposed_v) {
    constexpr int batch = 1;
    constexpr int heads = 4;
    constexpr int seq_q = 17;
    constexpr int seq_kv = 96;
    constexpr int head_size = 64;

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "SDPA micro requires IMMAD support";

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto q_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_q * head_size, -1.0f, 1.0f);
    const auto k_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_kv * head_size, -1.0f, 1.0f);
    const auto v_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_kv * head_size, -1.0f, 1.0f);

    std::vector<ov::float16> transposed_v_data(v_data.size());
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_kv; ++s) {
                for (int d = 0; d < head_size; ++d) {
                    const size_t src_idx = ((static_cast<size_t>(b) * heads + h) * seq_kv + s) * head_size + d;
                    const size_t dst_idx = ((static_cast<size_t>(b) * heads + h) * head_size + d) * seq_kv + s;
                    transposed_v_data[dst_idx] = v_data[src_idx];
                }
            }
        }
    }

    const layout q_layout({batch, heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout k_layout({batch, heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout v_layout({batch, heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout transposed_v_layout({batch, heads, head_size, seq_kv}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto v_mem = engine.allocate_memory(v_layout);
    auto transposed_v_mem = engine.allocate_memory(transposed_v_layout);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(transposed_v_mem, transposed_v_data);

    auto execute = [&](const layout& input_v_layout,
                       const memory::ptr& input_v,
                       const std::vector<int64_t>& input_v_order) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", input_v_layout));
        topo.add(scaled_dot_product_attention("sdpa",
                                               {input_info("q"), input_info("k"), input_info("v")},
                                               true,
                                               -1,
                                               {0, 1, 2, 3},
                                               {0, 1, 2, 3},
                                               input_v_order,
                                               {0, 1, 2, 3},
                                               {},
                                               false));
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::force_implementations(
            ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_micro"}}}));

        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

        std::string sdpa_info;
        for (const auto& info : network->get_primitives_info()) {
            if (info.type_id == "scaled_dot_product_attention") {
                sdpa_info = network->get_primitive_info(info.original_id);
                break;
            }
        }
        EXPECT_NE(sdpa_info.find("sdpa_micro"), std::string::npos)
            << "sdpa_micro was not selected; node description was:\n" << sdpa_info;

        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", input_v);
        return network->execute().at("result").get_memory();
    };

    auto reference_mem = execute(v_layout, v_mem, {0, 1, 2, 3});
    auto transposed_mem = execute(transposed_v_layout, transposed_v_mem, {0, 1, 3, 2});

    mem_lock<ov::float16, mem_lock_type::read> reference(reference_mem, get_test_stream());
    mem_lock<ov::float16, mem_lock_type::read> transposed(transposed_mem, get_test_stream());
    ASSERT_EQ(reference.size(), transposed.size());
    ASSERT_GE(cosineSimilarity(reference, transposed), 0.999f);
    for (size_t i = 0; i < reference.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(reference[i]), static_cast<float>(transposed[i]), 0.02f) << "Mismatch at index " << i;
    }
}

// Uses seq_q == seq_kv so the last query workgroup's causal_k always spans the full seq_kv,
// forcing the K/V loop to cross multiple K-tiles and exercise the TRANSPOSE_V advancement path.
TEST(sdpa_gpu_micro, transposed_v_matches_non_transposed_v_multi_tile) {
    constexpr int batch = 1;
    constexpr int heads = 4;
    constexpr int seq_q = 96;
    constexpr int seq_kv = 96;
    constexpr int head_size = 64;

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "SDPA micro requires IMMAD support";

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto q_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_q * head_size, -1.0f, 1.0f);
    const auto k_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_kv * head_size, -1.0f, 1.0f);
    const auto v_data = rg.generate_random_1d<ov::float16>(batch * heads * seq_kv * head_size, -1.0f, 1.0f);

    std::vector<ov::float16> transposed_v_data(v_data.size());
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_kv; ++s) {
                for (int d = 0; d < head_size; ++d) {
                    const size_t src_idx = ((static_cast<size_t>(b) * heads + h) * seq_kv + s) * head_size + d;
                    const size_t dst_idx = ((static_cast<size_t>(b) * heads + h) * head_size + d) * seq_kv + s;
                    transposed_v_data[dst_idx] = v_data[src_idx];
                }
            }
        }
    }

    const layout q_layout({batch, heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout k_layout({batch, heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout v_layout({batch, heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout transposed_v_layout({batch, heads, head_size, seq_kv}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto v_mem = engine.allocate_memory(v_layout);
    auto transposed_v_mem = engine.allocate_memory(transposed_v_layout);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(transposed_v_mem, transposed_v_data);

    auto execute = [&](const layout& input_v_layout,
                       const memory::ptr& input_v,
                       const std::vector<int64_t>& input_v_order) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", input_v_layout));
        topo.add(scaled_dot_product_attention("sdpa",
                                               {input_info("q"), input_info("k"), input_info("v")},
                                               true,
                                               -1,
                                               {0, 1, 2, 3},
                                               {0, 1, 2, 3},
                                               input_v_order,
                                               {0, 1, 2, 3},
                                               {},
                                               false));
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::force_implementations(
            ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_micro"}}}));

        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

        std::string sdpa_info;
        for (const auto& info : network->get_primitives_info()) {
            if (info.type_id == "scaled_dot_product_attention") {
                sdpa_info = network->get_primitive_info(info.original_id);
                break;
            }
        }
        EXPECT_NE(sdpa_info.find("sdpa_micro"), std::string::npos)
            << "sdpa_micro was not selected; node description was:\n" << sdpa_info;

        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", input_v);
        return network->execute().at("result").get_memory();
    };

    auto reference_mem = execute(v_layout, v_mem, {0, 1, 2, 3});
    auto transposed_mem = execute(transposed_v_layout, transposed_v_mem, {0, 1, 3, 2});

    mem_lock<ov::float16, mem_lock_type::read> reference(reference_mem, get_test_stream());
    mem_lock<ov::float16, mem_lock_type::read> transposed(transposed_mem, get_test_stream());
    ASSERT_EQ(reference.size(), transposed.size());
    ASSERT_GE(cosineSimilarity(reference, transposed), 0.999f);
    for (size_t i = 0; i < reference.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(reference[i]), static_cast<float>(transposed[i]), 0.02f) << "Mismatch at index " << i;
    }
}


// ---------------------------------------------------------------------------
// Compressed (int8 / int4) KV-cache SDPA tests.
//
// The optimized SDPA kernel (sdpa_opt) dequantizes the KV cache on read using
// the asymmetric formula `deq = (q - zp) * scale`, where a single (scale, zp)
// pair is shared by all head_size channels of a given (batch, head, token) and
// stored interleaved as [scale, zp] (InterleavedScalesZP) in fp16. INT4 packs
// two unsigned nibbles per byte: low nibble -> even head dim, high nibble -> odd
// head dim.
//
// These tests build a `scaled_dot_product_attention` primitive with
// is_kv_compressed=true directly and feed host-quantized KV plus the matching
// scale/zp buffers to the forced `sdpa_opt` path. The golden reference is an
// independent float attention (uncompressed, forced `sdpa_ref`) fed the exact
// host-dequantized KV, so the two networks operate on identical effective KV
// and any divergence isolates the kernel's dequant + attention math.
// ---------------------------------------------------------------------------
struct kv_quant_result {
    std::vector<int8_t> packed;            // quantized (INT4: two nibbles per byte)
    std::vector<ov::float16> dequantized;  // (q - zp) * scale, full head_size
    std::vector<ov::float16> scales;
    std::vector<ov::float16> zero_points;
};

enum class kv_quant_granularity {
    per_channel,
    per_token,
};

// Per-(batch, head, token) quantization over the whole head_size dim.
// Layout of `src` is bfyx {batch, seq, heads, head_size} (matches transpose {0,2,1,3}).
// Load a raw little-endian binary dump into a typed vector (no header parsing).
template <typename T>
static std::vector<T> load_bin_as(const std::string& path) {
    const auto bytes = ov::util::load_binary(path);
    OPENVINO_ASSERT(!bytes.empty(), "Failed to load (missing/empty) binary file: ", path);
    OPENVINO_ASSERT(bytes.size() % sizeof(T) == 0, "Binary file size is not a multiple of element size: ", path);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

// Per-token KV quantization: one (scale, zp) per (batch, head, token), covering the
// whole head dimension. When seq_major is true the source is laid out
// [batch, seq, heads, head_size]; when false it is [batch, heads, seq, head_size]. The produced
// scales / zero_points buffers are always laid out [batch, heads, seq].
//
// Two quantization strategies are supported:
//   * Symmetric  (symmetric=true):  signed codes centered on zero, deq = q * scale (no
//     zero-point). The per-token scale is max_abs / q_max over the head dimension.
//   * Asymmetric (symmetric=false): unsigned codes with a zero-point, deq = (q - zp) * scale.
//     The per-token scale/zp are derived from the [min, max] range over the head dimension
//     positions: scale = (max - min) / (q_max - q_min), zp = q_min - min / scale.
// In both cases the derived scale is multiplied by a per-token random factor (see below).
// provided_scale, when non-null, supplies the per-token scales ([batch, heads, seq])
// instead of deriving (and randomizing) them; the zero-point is still derived for the asymmetric
// case so it matches the given scale.
static kv_quant_result quantize_kv_per_token(const std::vector<ov::float16>& src,
                                               int batch,
                                               int seq,
                                               int heads,
                                               int head_size,
                                               int bit_width,
                                               bool seq_major = true,
                                               const std::vector<ov::float16>* provided_scale = nullptr,
                                               bool symmetric = false) {
    // Symmetric uses a signed range centered on zero (no zero-point); asymmetric int4 uses
    // unsigned nibbles [0, 15] with a zero-point.
    const int q_min = symmetric ? -(bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 0 : -128);
    const int q_max = symmetric ? (bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 15 : 127);
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;
    const size_t token_count = static_cast<size_t>(batch) * heads * seq;

    kv_quant_result r;
    r.packed.assign(static_cast<size_t>(batch) * seq * heads * packed_hs, 0);
    r.dequantized.assign(static_cast<size_t>(batch) * seq * heads * head_size, ov::float16(0.0f));
    r.scales.assign(token_count, ov::float16(0.0f));
    if (!symmetric)
        r.zero_points.assign(token_count, ov::float16(0.0f));

    const auto elem_base = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * head_size : ((static_cast<size_t>(b) * heads + h) * seq + s) * head_size;
    };
    const auto packed_base_of = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * packed_hs : ((static_cast<size_t>(b) * heads + h) * seq + s) * packed_hs;
    };
    const auto pack_value = [&](size_t packed_base, int d, int q) {
        if (bit_width == 4) {
            // low nibble = even head dim, high nibble = odd head dim
            auto& byte = r.packed[packed_base + d / 2];
            if (d % 2 == 0)
                byte = static_cast<int8_t>((byte & 0xF0) | (q & 0x0F));
            else
                byte = static_cast<int8_t>((byte & 0x0F) | ((q & 0x0F) << 4));
        } else {
            r.packed[packed_base + d] = static_cast<int8_t>(q);
        }
    };

    // Random per-token multiplier applied on top of the derived scale, generated the same way as
    // the tests' random inputs (1/8-resolution values in [1, 2]). Neighbouring tokens therefore
    // get clearly different scales, so a kernel that mis-indexes the scale buffer diverges from the
    // host dequantization instead of silently agreeing; staying >= 1 keeps every value inside the
    // quantized range. The fixed seed keeps the test reproducible.
    tests::random_generator scale_rg("quantize_kv_per_token_scale");
    const auto scale_mul = scale_rg.generate_random_1d<float>(token_count, 1, 2, 8);

    // Derive (or take) one scale and zero-point per (batch, head, token) from the values of
    // that token across the whole head.
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq; ++s) {
                const size_t token = (static_cast<size_t>(b) * heads + h) * seq + s;
                float min_v = std::numeric_limits<float>::max();
                float max_v = std::numeric_limits<float>::lowest();
                for (int d = 0; d < head_size; ++d) {
                    const float v = static_cast<float>(src[elem_base(b, s, h) + d]);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }
                float scale = 0.0f;
                if (provided_scale != nullptr) {
                    scale = static_cast<float>((*provided_scale)[token]);
                } else if (symmetric) {
                    scale = std::max(std::abs(min_v), std::abs(max_v)) / static_cast<float>(q_max) * scale_mul[token];
                } else {
                    scale = (max_v - min_v) / static_cast<float>(q_max - q_min) * scale_mul[token];
                }
                if (scale == 0.0f)
                    scale = 1.0f;  // degenerate (constant) token: avoid div-by-zero
                r.scales[token] = ov::float16(scale);
                if (!symmetric) {
                    // deq = (q - zp) * scale, so min_v maps to q_min: zp = q_min - min_v / scale.
                    r.zero_points[token] = ov::float16(std::round(static_cast<float>(q_min) - min_v / scale));
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = elem_base(b, s, h);
                const size_t packed_base = packed_base_of(b, s, h);
                const size_t scale_base = (static_cast<size_t>(b) * heads + h) * seq + s;
                for (int d = 0; d < head_size; ++d) {
                    // Quantize through the fp16-rounded scale/zp so the host dequantization
                    // matches exactly what the kernel reads from the scale/zp buffers.
                    const float scale = static_cast<float>(r.scales[scale_base]);
                    const float zp = symmetric ? 0.0f : static_cast<float>(r.zero_points[scale_base]);
                    const float v = static_cast<float>(src[base + d]);
                    int q = static_cast<int>(std::lround(v / scale + zp));
                    q = std::max(q_min, std::min(q_max, q));
                    r.dequantized[base + d] = ov::float16((static_cast<float>(q) - zp) * scale);
                    pack_value(packed_base, d, q);
                }
            }
        }
    }
    return r;
}

// Per-channel KV quantization: one scale/zp per (batch, head, channel), shared by all tokens.
// The compression buffers are laid out as [batch, heads, 1, head_size].
static kv_quant_result quantize_kv_per_channel(const std::vector<ov::float16>& src,
                                               int batch,
                                               int seq,
                                               int heads,
                                               int head_size,
                                               int bit_width,
                                               bool seq_major = true,
                                               const std::vector<ov::float16>* provided_scale = nullptr,
                                               bool symmetric = false) {
    const int q_min = symmetric ? -(bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 0 : -128);
    const int q_max = symmetric ? (bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 15 : 127);
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;
    const size_t channel_count = static_cast<size_t>(batch) * heads * head_size;

    kv_quant_result r;
    r.packed.assign(static_cast<size_t>(batch) * seq * heads * packed_hs, 0);
    r.dequantized.assign(static_cast<size_t>(batch) * seq * heads * head_size, ov::float16(0.0f));
    r.scales.assign(channel_count, ov::float16(0.0f));
    if (!symmetric)
        r.zero_points.assign(channel_count, ov::float16(0.0f));

    const auto elem_base = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * head_size
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * head_size;
    };
    const auto packed_base_of = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * packed_hs
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * packed_hs;
    };
    const auto pack_value = [&](size_t packed_base, int d, int q) {
        if (bit_width == 4) {
            auto& byte = r.packed[packed_base + d / 2];
            if (d % 2 == 0)
                byte = static_cast<int8_t>((byte & 0xF0) | (q & 0x0F));
            else
                byte = static_cast<int8_t>((byte & 0x0F) | ((q & 0x0F) << 4));
        } else {
            r.packed[packed_base + d] = static_cast<int8_t>(q);
        }
    };

    tests::random_generator scale_rg("quantize_kv_per_channel_scale");
    const auto scale_mul = scale_rg.generate_random_1d<float>(channel_count, 1, 2, 8);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < head_size; ++d) {
                const size_t channel = (static_cast<size_t>(b) * heads + h) * head_size + d;
                float min_v = std::numeric_limits<float>::max();
                float max_v = std::numeric_limits<float>::lowest();
                for (int s = 0; s < seq; ++s) {
                    const float v = static_cast<float>(src[elem_base(b, s, h) + d]);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }

                float scale = 0.0f;
                if (provided_scale != nullptr) {
                    scale = static_cast<float>((*provided_scale)[channel]);
                } else if (symmetric) {
                    scale = std::max(std::abs(min_v), std::abs(max_v)) / static_cast<float>(q_max) *
                            scale_mul[channel];
                } else {
                    scale = (max_v - min_v) / static_cast<float>(q_max - q_min) * scale_mul[channel];
                }
                if (scale == 0.0f)
                    scale = 1.0f;

                r.scales[channel] = ov::float16(scale);
                if (!symmetric)
                    r.zero_points[channel] =
                        ov::float16(std::round(static_cast<float>(q_min) - min_v / scale));
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = elem_base(b, s, h);
                const size_t packed_base = packed_base_of(b, s, h);
                const size_t scale_base = (static_cast<size_t>(b) * heads + h) * head_size;
                for (int d = 0; d < head_size; ++d) {
                    const float scale = static_cast<float>(r.scales[scale_base + d]);
                    const float zp = symmetric ? 0.0f : static_cast<float>(r.zero_points[scale_base + d]);
                    const float v = static_cast<float>(src[base + d]);
                    int q = static_cast<int>(std::lround(v / scale + zp));
                    q = std::max(q_min, std::min(q_max, q));
                    r.dequantized[base + d] = ov::float16((static_cast<float>(q) - zp) * scale);
                    pack_value(packed_base, d, q);
                }
            }
        }
    }

    return r;
}

// Per-channel symmetric quantization that reproduces the GQA decomposition's write path.
// INT4 values use a +8 storage bias and are exposed to SDPA as asymmetric u4 with zero point 8.
// INT8 values retain their signed representation and are exposed as symmetric i8.
static kv_quant_result quantize_kv_per_channel_gqa_decomp(const std::vector<ov::float16>& src,
                                                          int batch,
                                                          int seq,
                                                          int heads,
                                                          int head_size,
                                                          int bit_width,
                                                          bool seq_major = true,
                                                          const std::vector<ov::float16>* provided_scale = nullptr,
                                                          bool symmetric = false) {
    OPENVINO_ASSERT(bit_width == 4 || bit_width == 8, "GQA decomposition supports only INT4 and INT8 KV caches");
    OPENVINO_ASSERT(symmetric, "GQA decomposition uses symmetric KV quantization");
    OPENVINO_ASSERT(bit_width != 4 || head_size % 2 == 0, "INT4 KV packing requires an even head size");
    const int q_min = bit_width == 4 ? -8 : -128;
    const int q_max = bit_width == 4 ? 7 : 127;
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;
    const size_t channel_count = static_cast<size_t>(batch) * heads * head_size;

    kv_quant_result r;
    r.packed.assign(static_cast<size_t>(batch) * seq * heads * packed_hs, 0);
    r.dequantized.assign(static_cast<size_t>(batch) * seq * heads * head_size, ov::float16(0.0f));
    r.scales.assign(channel_count, ov::float16(0.0f));
    if (bit_width == 4)
        r.zero_points.assign(channel_count, ov::float16(8.0f));

    const auto elem_base = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * head_size
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * head_size;
    };
    const auto packed_base_of = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * packed_hs
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * packed_hs;
    };

    tests::random_generator scale_rg("quantize_kv_per_channel_gqa_decomp_scale");
    const auto scale_mul = scale_rg.generate_random_1d<float>(channel_count, 1, 2, 8);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < head_size; ++d) {
                const size_t channel = (static_cast<size_t>(b) * heads + h) * head_size + d;
                float min_v = std::numeric_limits<float>::max();
                float max_v = std::numeric_limits<float>::lowest();
                for (int s = 0; s < seq; ++s) {
                    const float v = static_cast<float>(src[elem_base(b, s, h) + d]);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }

                float scale = 0.0f;
                if (provided_scale != nullptr) {
                    scale = static_cast<float>((*provided_scale)[channel]);
                } else {
                    scale = std::max(std::abs(min_v), std::abs(max_v)) / static_cast<float>(q_max) * scale_mul[channel];
                }
                if (scale == 0.0f)
                    scale = 1.0f;

                r.scales[channel] = ov::float16(scale);
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = elem_base(b, s, h);
                const size_t packed_base = packed_base_of(b, s, h);
                const size_t scale_base = (static_cast<size_t>(b) * heads + h) * head_size;
                for (int d = 0; d < head_size; ++d) {
                    const float scale = static_cast<float>(r.scales[scale_base + d]);
                    const float v = static_cast<float>(src[base + d]);
                    int q = static_cast<int>(std::nearbyint(v / scale));
                    q = std::max(q_min, std::min(q_max, q));
                    r.dequantized[base + d] = ov::float16(static_cast<float>(q) * scale);
                    if (bit_width == 4) {
                        const int nibble = q + 8;
                        auto& byte = r.packed[packed_base + d / 2];
                        if (d % 2 == 0)
                            byte = static_cast<int8_t>((byte & 0xF0) | (nibble & 0x0F));
                        else
                            byte = static_cast<int8_t>((byte & 0x0F) | ((nibble & 0x0F) << 4));
                    } else {
                        r.packed[packed_base + d] = static_cast<int8_t>(q);
                    }
                }
            }
        }
    }

    return r;
}

static void run_compressed_kv_sdpa_test(const sdpa_test_params& params,
                                        kv_quant_granularity granularity,
                                        bool gqa_decomp = false) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (granularity == kv_quant_granularity::per_channel) {
        const auto& device_info = engine.get_device_info();
        if (!device_info.supports_immad || device_info.arch < gpu_arch::xe_hpg) {
            GTEST_SKIP() << "Per-channel compressed KV is supported by micro SDPA only";
        }
    }

    const int bit_width = params.bit_width;
    const bool asymmetric = params.asymmetric;
    const bool has_storage_zero_point = asymmetric || (gqa_decomp && bit_width == 4);
    const int batch = params.batch;
    const int q_num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const int seq_q = params.sequence_length_q;
    const int seq_kv = params.sequence_length_kv;
    const int head_size = params.head_size;

    OPENVINO_ASSERT(bit_width != 4 || head_size % 2 == 0, "INT4 KV packing requires an even head size");
    const int packed_head_size = bit_width == 4 ? head_size / 2 : head_size;

    // Q and original float K/V use [batch, heads, sequence, head_size].
    auto q_data = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_q * q_num_heads * head_size, -1.0f, 1.0f);
    auto k_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * kv_num_heads * head_size, -1.0f, 1.0f);
    auto v_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * kv_num_heads * head_size, -1.0f, 1.0f);

    const auto quantize_kv = [&]() {
        if (gqa_decomp) {
            OPENVINO_ASSERT(granularity == kv_quant_granularity::per_channel && !asymmetric,
                            "GQA decomposition requires symmetric per-channel quantization");
            return &quantize_kv_per_channel_gqa_decomp;
        }
        return granularity == kv_quant_granularity::per_channel ? quantize_kv_per_channel
                                                                : quantize_kv_per_token;
    }();
    auto k_q = quantize_kv(k_orig,
                                       batch,
                                       seq_kv,
                                       kv_num_heads,
                                       head_size,
                                       bit_width,
                                       /*seq_major=*/false,
                                       nullptr,
                                       /*symmetric=*/!asymmetric);
    auto v_q = quantize_kv(v_orig,
                                       batch,
                                       seq_kv,
                                       kv_num_heads,
                                       head_size,
                                       bit_width,
                                       /*seq_major=*/false,
                                       nullptr,
                                       /*symmetric=*/!asymmetric);

    const layout q_layout({batch, q_num_heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout kv_deq_layout({batch, kv_num_heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    // INT4 stores two adjacent head-dimension values in each byte: [B, H, S, D/2].
    const layout kv_packed_layout({batch, kv_num_heads, seq_kv, packed_head_size}, data_types::i8, format::bfyx);
    const layout comp_layout(granularity == kv_quant_granularity::per_channel
                                 ? ov::PartialShape{batch, kv_num_heads, 1, head_size}
                                 : ov::PartialShape{batch, kv_num_heads, seq_kv, 1},
                             data_types::f16,
                             format::bfyx);

    const ov::Shape expected_packed_shape = {static_cast<size_t>(batch),
                                             static_cast<size_t>(kv_num_heads),
                                             static_cast<size_t>(seq_kv),
                                             static_cast<size_t>(packed_head_size)};
    ASSERT_EQ(kv_packed_layout.get_shape(), expected_packed_shape);
    ASSERT_EQ(k_q.packed.size(), ov::shape_size(expected_packed_shape));
    ASSERT_EQ(v_q.packed.size(), ov::shape_size(expected_packed_shape));

    auto q_mem = engine.allocate_memory(q_layout);
    set_values(q_mem, q_data);

    // --- Golden reference: uncompressed float attention on host-dequantized KV (sdpa_ref) ---
    auto make_ref_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_deq_layout);
        auto v_mem = engine.allocate_memory(kv_deq_layout);
        set_values(k_mem, k_q.dequantized);
        set_values(v_mem, v_q.dequantized);

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_deq_layout));
        topo.add(input_layout("v", kv_deq_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v")},
                                                 true,
                                                 -1,
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {},
                                                 false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Compressed path: quantized KV + interleaved scale/zp on the optimized kernel (sdpa_opt) ---
    auto make_opt_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_packed_layout);
        auto v_mem = engine.allocate_memory(kv_packed_layout);
        auto k_comp_mem = engine.allocate_memory(comp_layout);
        auto v_comp_mem = engine.allocate_memory(comp_layout);
        auto k_zp_mem = has_storage_zero_point ? engine.allocate_memory(comp_layout) : nullptr;
        auto v_zp_mem = has_storage_zero_point ? engine.allocate_memory(comp_layout) : nullptr;
        set_values(k_mem, k_q.packed);
        set_values(v_mem, v_q.packed);
        set_values(k_comp_mem, k_q.scales);
        set_values(v_comp_mem, v_q.scales);
        if (has_storage_zero_point) {
            set_values(k_zp_mem, k_q.zero_points);
            set_values(v_zp_mem, v_q.zero_points);
        }

        scaled_dot_product_attention::QuantizationAttributes qa;
        qa.quantization_type = has_storage_zero_point ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                                                     : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        qa.quantization_dt = bit_width == 4 ? (has_storage_zero_point ? ov::element::u4 : ov::element::i4) : ov::element::i8;
        qa.scale_dt = ov::element::f16;
        qa.zp_dt = ov::element::f16;
        qa.group_sizes = granularity == kv_quant_granularity::per_channel
                             ? std::vector<uint64_t>{1, 1, UINT64_MAX, 1}
                             : std::vector<uint64_t>{1, 1, 1, UINT64_MAX};
        qa.scales_zp_output_order = {0, 1, 2, 3};
        qa.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_packed_layout));
        topo.add(input_layout("v", kv_packed_layout));
        topo.add(input_layout("k_scale", comp_layout));
        topo.add(input_layout("v_scale", comp_layout));
        std::vector<input_info> inputs = {input_info("q"), input_info("k"), input_info("v"), input_info("k_scale"), input_info("v_scale")};
        if (has_storage_zero_point) {
            topo.add(input_layout("k_zp", comp_layout));
            topo.add(input_layout("v_zp", comp_layout));
            inputs.emplace_back("k_zp");
            inputs.emplace_back("v_zp");
        }

        auto prim = scaled_dot_product_attention("sdpa", inputs, true, -1, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, qa, true);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::hint::kv_cache_precision(bit_width == 4 ? ov::element::i4 : ov::element::i8));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("k_scale", k_comp_mem);
        net->set_input_data("v_scale", v_comp_mem);
        if (has_storage_zero_point) {
            net->set_input_data("k_zp", k_zp_mem);
            net->set_input_data("v_zp", v_zp_mem);
        }
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < opt_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in compressed output at index " << i;
    }
    const float sim = cosineSimilarity(ref_ptr, opt_ptr);
    ASSERT_GE(sim, 0.999f) << "bit_width=" << bit_width << ", asymmetric=" << asymmetric << " cosine similarity too low: " << sim;
}

struct sdpa_gpu_compressed_kv_test_base : public ::testing::TestWithParam<sdpa_test_params> {
    static std::string PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        const auto& p = info.param;
        return std::string("int") + std::to_string(p.bit_width) + (p.asymmetric ? "_asymmetric_" : "_symmetric_") +
               (p.num_heads == p.kv_num_heads ? "mha_" : "gqa_") + (p.sequence_length_q == 1 ? "decode" : "prefill");
    }
};

struct sdpa_gpu_compressed_kv_per_channel_test : public sdpa_gpu_compressed_kv_test_base {};

TEST_P(sdpa_gpu_compressed_kv_per_channel_test, compare_with_gpu_dequantized_reference) {
    run_compressed_kv_sdpa_test(GetParam(), kv_quant_granularity::per_channel);
}

struct sdpa_gpu_compressed_kv_per_token_test : public sdpa_gpu_compressed_kv_test_base {};

TEST_P(sdpa_gpu_compressed_kv_per_token_test, compare_with_gpu_dequantized_reference) {
    run_compressed_kv_sdpa_test(GetParam(), kv_quant_granularity::per_token);
}

// Verify the INT4 and INT8 cache encodings produced by the GQA decomposition.
struct sdpa_gpu_gqa_decomp_test : public sdpa_gpu_compressed_kv_test_base {};

TEST_P(sdpa_gpu_gqa_decomp_test, compare_with_gpu_dequantized_reference) {
    run_compressed_kv_sdpa_test(GetParam(), kv_quant_granularity::per_channel, /*gqa_decomp=*/true);
}

INSTANTIATE_TEST_SUITE_P(smoke_sdpa_gpu_compressed_kv_per_channel,
                         sdpa_gpu_compressed_kv_per_channel_test,
                         ::testing::Values(sdpa_test_params{128, 40, 10, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, true}),
                         sdpa_gpu_compressed_kv_per_channel_test::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_sdpa_gpu_compressed_kv_per_token,
                         sdpa_gpu_compressed_kv_per_token_test,
                         ::testing::Values(sdpa_test_params{128, 40, 10, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, true}),
                         sdpa_gpu_compressed_kv_per_token_test::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_sdpa_gpu_gqa_decomp,
                         sdpa_gpu_gqa_decomp_test,
                         ::testing::Values(sdpa_test_params{128, 40, 10, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, false}),
                         sdpa_gpu_gqa_decomp_test::PrintToStringParamName);
// Which DPAS kernel plain SDPA dispatches: sdpa_ocl on Xe2+ XMX, sdpa_micro on the other XMX parts (as upstream),
// the opt kernels elsewhere -- one choice per device, with no fallback from one DPAS kernel to the other. Dispatch
// only: this harness's "sdpa_ref" network also runs SDPAOptImpl (force_implementations keeps only the impl type), so
// accuracy is left to the functional ScaledAttn tests. Static shapes; the second test rebuilds the impl through
// program save/load (default ctor plus the saved stage order), so the dispatch must not depend on state only the
// params ctor had.
class sdpa_dpas_backend_test : public sdpa_gpu_test {
public:
    void check_dispatch(bool is_caching_test) {
        auto p = GetParam();
        auto& engine = get_test_engine();
        const auto& info = engine.get_device_info();
        const bool prefill = p.sequence_length_q > 1;
        const bool unaligned = p.head_size % 16 != 0;
        const bool is_ARL_H = info.gfx_ver.major == 12 && info.gfx_ver.minor == 74;
        // This harness always feeds a mask input, which is PLAIN_EXT on xe_hpg (SDPAOclGenerator::hpg_tier_required()).
        const auto backend = tests::expected_dpas_backend(engine, false, static_cast<size_t>(p.head_size),
                                                          ov::intel_gpu::ocl::PLAIN_F16_STATIC | ov::intel_gpu::ocl::PLAIN_EXT);
        std::string expected;
        switch (backend) {
        case tests::dpas_backend::ocl:
            // The sdpa_ocl single-token kernel also takes unaligned heads.
            expected = prefill ? "sdpa_ocl_prefill" : "sdpa_ocl_mixed";
            break;
        case tests::dpas_backend::micro:
            // sdpa_micro single-token does not take unaligned heads, and ARL-H keeps the opt one for static decode.
            expected = prefill ? "sdpa_micro__prefill" : unaligned ? "sdpa_opt__multi_reg" : is_ARL_H ? "sdpa_opt__single_reg" : "sdpa_micro__generate";
            break;
        case tests::dpas_backend::none:
            expected = (prefill || unaligned) ? "sdpa_opt__multi_reg" : "sdpa_opt__single_reg";
            break;
        }

        const auto q_layout = cldnn::layout({p.batch, p.sequence_length_q, p.num_heads, p.head_size}, p.dt, format::bfyx);
        const auto kv_layout = cldnn::layout({p.batch, p.sequence_length_kv, p.num_heads, p.head_size}, p.dt, format::bfyx);
        const auto mask_layout = p.sequence_length_q == p.sequence_length_kv
                                     ? cldnn::layout({p.sequence_length_q, p.sequence_length_kv}, p.dt, format::bfyx)
                                     : cldnn::layout({p.batch, p.num_heads, 1, p.sequence_length_kv}, p.dt, format::bfyx);
        auto q = engine.allocate_memory(q_layout);
        auto k = engine.allocate_memory(kv_layout);
        auto v = engine.allocate_memory(kv_layout);
        auto mask = engine.allocate_memory(mask_layout);
        load_input(q, 0, p.dt);
        load_input(k, 1, p.dt);
        load_input(v, 2, p.dt);
        load_input(mask, 3, p.dt);

        auto [output, net] = run_network(is_caching_test, true, q_layout, kv_layout, kv_layout, mask_layout, q, k, v, mask,
                                         false, 1.0f, false, 0.0f, p.dt);
        std::shared_ptr<cldnn::primitive_inst> sdpa_inst;
        for (const auto& prim_info : net->get_primitives_info()) {
            if (prim_info.type_id == "scaled_dot_product_attention")
                sdpa_inst = net->get_primitive(prim_info.original_id);
        }
        ASSERT_NE(sdpa_inst, nullptr);
        ASSERT_NE(sdpa_inst->get_impl(), nullptr);
        const auto entries = sdpa_inst->get_impl()->get_kernels_dump_info(*sdpa_inst->get_impl_params()).get_entries();
        EXPECT_NE(entries.find(expected), std::string::npos) << "expected " << expected << ", dispatched: " << entries;
        if (backend != tests::dpas_backend::ocl) {
            EXPECT_EQ(entries.find("sdpa_ocl"), std::string::npos) << "dispatched: " << entries;
        }
        if (backend != tests::dpas_backend::micro) {
            EXPECT_EQ(entries.find("sdpa_micro"), std::string::npos) << "dispatched: " << entries;
        }

        cldnn::mem_lock<ov::float16, mem_lock_type::read> out_data(output, get_test_stream());
        for (size_t idx = 0; idx < out_data.size(); idx++) {
            ASSERT_FALSE(std::isnan(out_data[idx])) << "NaN found at index " << idx;
        }
    }
};

TEST_P(sdpa_dpas_backend_test, dispatches_lane_kernel) {
    check_dispatch(false);
}

TEST_P(sdpa_dpas_backend_test, dispatches_lane_kernel_after_load) {
    check_dispatch(true);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_dpas_backend_selection,
    sdpa_dpas_backend_test,
    ::testing::Values(
        sdpa_test_params{64, 32, 128, 128, 2, false},  // static prefill
        sdpa_test_params{64, 32, 1, 128, 2, false},    // static decode
        sdpa_test_params{72, 8, 1, 128, 2, false}      // static decode, head size % 16 != 0
    ),
    sdpa_gpu_test::PrintToStringParamName
);

// Sharp-softmax check of the SG8 (xe_hpg) operand mapping of sdpa_ocl, plain f16, static, q == kv > 1, no mask / causal / scale input
// (the PLAIN_F16_STATIC tier). Every query's logit row has ONE key that wins by a wide margin, and every key / value row is unique, so
// a lane that reads the wrong head dim, key pair or query row moves the winner or mixes in another value row: the output is off by
// ~the value spread, far above the f16 tolerance. Random N(0, 0.1)-like data would not show it (pa-harness-data-hides-qk-errors).
// Runs wherever sdpa_ocl serves the op (Xe2 and xe_hpg with TEST_USE_SDPA_OCL_HPG=1), SKIPs elsewhere; the SDPA_OCL_NEG_SG8=1..3
// groups of test/sdpa_ocl_gtests.sh break the mapping on purpose and must FAIL it. max_abs_err is recorded so a NEG run that fails on
// the error can be told from one that fails on a crash.
struct sdpa_hpg_sharp_params {
    int head_size;
    int seq_len;
};

struct sdpa_hpg_sharp_test : public ::testing::TestWithParam<sdpa_hpg_sharp_params> {
    static constexpr int num_heads = 8;
    static constexpr float key_gain = 32.0f;  // Q = key_gain * K[winner]: the scaled winner logit leads by >= ~20 (head 32) .. ~2500 (head 256)

    // A deterministic pseudo-random value in [-15/16, 15/16] on a 1/16 grid (exact in f16), distinct per (stream, row, col, head).
    // The modulus is a prime, so no row repeats with a short period.
    static float grid_value(uint32_t stream, uint32_t head, uint32_t row, uint32_t col) {
        uint32_t h = stream * 0x9E3779B1u ^ (head * 0x85EBCA6Bu + 0x27D4EB2Fu) ^ (row * 73856093u) ^ (col * 19349663u + 83492791u);
        h ^= h >> 13;
        h *= 1274126177u;
        h ^= h >> 16;
        return (static_cast<float>(h % 31u) - 15.0f) / 16.0f;
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<sdpa_hpg_sharp_params>& info) {
        return "head" + std::to_string(info.param.head_size) + "_seq" + std::to_string(info.param.seq_len);
    }
};

TEST_P(sdpa_hpg_sharp_test, f16_static_prefill) {
    const auto p = GetParam();
    auto& engine = get_test_engine();
    if (tests::expected_dpas_backend(engine, false, static_cast<size_t>(p.head_size)) != tests::dpas_backend::ocl)
        GTEST_SKIP() << "sdpa_ocl does not serve plain f16 static SDPA on this device (xe_hpg needs TEST_USE_SDPA_OCL_HPG=1 and TEST_USE_SDPA_OCL not 0)";

    const int d = p.head_size;
    const int n = p.seq_len;
    const size_t count = static_cast<size_t>(num_heads) * n * d;
    std::vector<ov::float16> q_data(count), k_data(count), v_data(count);
    std::vector<int> winner(n);
    for (int i = 0; i < n; ++i)
        winner[i] = static_cast<int>((17u * static_cast<uint32_t>(i) + 3u) % static_cast<uint32_t>(n));
    for (int h = 0; h < num_heads; ++h) {
        for (int j = 0; j < n; ++j) {
            for (int c = 0; c < d; ++c) {
                const size_t at = (static_cast<size_t>(h) * n + j) * d + c;
                k_data[at] = ov::float16(sdpa_hpg_sharp_test::grid_value(1, h, j, c));
                v_data[at] = ov::float16(sdpa_hpg_sharp_test::grid_value(2, h, j, c) * 2.0f);
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < d; ++c) {
                const size_t at = (static_cast<size_t>(h) * n + i) * d + c;
                q_data[at] = ov::float16(sdpa_hpg_sharp_test::key_gain * sdpa_hpg_sharp_test::grid_value(1, h, winner[i], c));
            }
        }
    }

    const layout layout_bhsd({1, num_heads, n, d}, data_types::f16, format::bfyx);
    auto q_mem = engine.allocate_memory(layout_bhsd);
    auto k_mem = engine.allocate_memory(layout_bhsd);
    auto v_mem = engine.allocate_memory(layout_bhsd);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);

    topology topo;
    topo.add(input_layout("q", layout_bhsd));
    topo.add(input_layout("k", layout_bhsd));
    topo.add(input_layout("v", layout_bhsd));
    topo.add(scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v")}, false, -1, {0, 1, 2, 3}, {0, 1, 2, 3},
                                          {0, 1, 2, 3}, {0, 1, 2, 3}, {}, false));
    topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
    net->set_input_data("q", q_mem);
    net->set_input_data("k", k_mem);
    net->set_input_data("v", v_mem);
    auto output = net->execute().at("result").get_memory();

    // The kernel that ran, not just the lane that was selected: the whole point is the sdpa_ocl SG8 arm.
    std::shared_ptr<cldnn::primitive_inst> sdpa_inst = net->get_primitive("sdpa");
    ASSERT_NE(sdpa_inst, nullptr);
    ASSERT_NE(sdpa_inst->get_impl(), nullptr);
    const auto entries = sdpa_inst->get_impl()->get_kernels_dump_info(*sdpa_inst->get_impl_params()).get_entries();
    ASSERT_NE(entries.find("sdpa_ocl_prefill"), std::string::npos) << "dispatched: " << entries;

    // CPU reference in double over the stored (f16-rounded) inputs, default scale 1 / sqrt(head).
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(output, get_test_stream());
    ASSERT_EQ(out_ptr.size(), count);
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    double max_abs_err = 0.0;
    std::vector<double> logit(n), acc(d);
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < n; ++i) {
            double max_logit = -INFINITY;
            for (int j = 0; j < n; ++j) {
                double dot = 0.0;
                for (int c = 0; c < d; ++c)
                    dot += static_cast<double>(static_cast<float>(q_data[(static_cast<size_t>(h) * n + i) * d + c])) *
                           static_cast<double>(static_cast<float>(k_data[(static_cast<size_t>(h) * n + j) * d + c]));
                logit[j] = dot * scale;
                max_logit = std::max(max_logit, logit[j]);
            }
            double sum = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < n; ++j) {
                const double w = std::exp(logit[j] - max_logit);
                sum += w;
                for (int c = 0; c < d; ++c)
                    acc[c] += w * static_cast<double>(static_cast<float>(v_data[(static_cast<size_t>(h) * n + j) * d + c]));
            }
            for (int c = 0; c < d; ++c) {
                const float got = static_cast<float>(out_ptr[(static_cast<size_t>(h) * n + i) * d + c]);
                ASSERT_FALSE(std::isnan(got)) << "NaN at head " << h << " query " << i << " col " << c;
                max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(got) - acc[c] / sum));
            }
        }
    }
    RecordProperty("max_abs_err", std::to_string(max_abs_err));
    EXPECT_LE(max_abs_err, 1e-2) << "head " << d << " seq " << n << " max_abs_err " << max_abs_err;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_hpg_sharp,
    sdpa_hpg_sharp_test,
    ::testing::Values(
        // seq 100 is not a multiple of any query / key tile; head 40 and 72 are not a multiple of the 16-deep DPAS step
        sdpa_hpg_sharp_params{32, 16},  sdpa_hpg_sharp_params{32, 64},  sdpa_hpg_sharp_params{32, 100},  sdpa_hpg_sharp_params{32, 1024},
        sdpa_hpg_sharp_params{40, 16},  sdpa_hpg_sharp_params{40, 64},  sdpa_hpg_sharp_params{40, 100},  sdpa_hpg_sharp_params{40, 1024},
        sdpa_hpg_sharp_params{64, 16},  sdpa_hpg_sharp_params{64, 64},  sdpa_hpg_sharp_params{64, 100},  sdpa_hpg_sharp_params{64, 1024},
        sdpa_hpg_sharp_params{72, 16},  sdpa_hpg_sharp_params{72, 64},  sdpa_hpg_sharp_params{72, 100},  sdpa_hpg_sharp_params{72, 1024},
        sdpa_hpg_sharp_params{96, 16},  sdpa_hpg_sharp_params{96, 64},  sdpa_hpg_sharp_params{96, 100},  sdpa_hpg_sharp_params{96, 1024},
        sdpa_hpg_sharp_params{128, 16}, sdpa_hpg_sharp_params{128, 64}, sdpa_hpg_sharp_params{128, 100}, sdpa_hpg_sharp_params{128, 1024},
        sdpa_hpg_sharp_params{256, 16}, sdpa_hpg_sharp_params{256, 64}, sdpa_hpg_sharp_params{256, 100}, sdpa_hpg_sharp_params{256, 1024}
    ),
    sdpa_hpg_sharp_test::PrintToStringParamName
);

// The rest of plain SDPA on xe_hpg (the PLAIN_EXT tier): bf16, attn_mask (per key / full 2D / const scalar), causal, sink, dynamic
// shape, a single query, and a K view whose row pitch or base breaks the 4 B alignment of the dword K read. Same idea as the sharp
// test: every value row is unique and the softmax has a clear winner per (head, query), so a lane that reads the wrong mask element,
// key pair or query row moves the winner and the output is off by ~the value spread, far above the tolerance. Two data modes,
// because one mode cannot show both kinds of error:
//   sharp : Q = 32 * K[winner]; the QK logit decides. A mask / sink here would be invisible (the logit gap dwarfs it).
//   masked: Q is small independent noise and the mask (or the mask plus the sink) decides: mask = -2.5 * |key - winner| puts a
//           weight of 1 on the winner, 0.08 on its neighbours, so a mask element read from the wrong key moves the winner. Softmax
//           is shift invariant, so a const scalar mask is only a does-it-run check (it uses sharp data).
// max_abs_err is recorded as in the sharp test; SDPA_OCL_NEG_SG8=5 / 6 break the full / per-key mask read on purpose.
struct sdpa_hpg_ext_params {
    int head_size;
    int q_len;
    int kv_len;
    bool bf16;
    int mask;     // 0 none, 1 per key [1, H, 1, kv], 2 full [1, H, q, kv], 3 const scalar (primitive attn_mask_val)
    int causal;   // 0 off, 1 top-left aligned, 2 lower-right aligned
    bool sink;    // adds the scale and sink inputs; needs mask 1 or 2
    bool dynamic;
    int k_pad;    // 0 none, 1 head dim padded (1, 1): even pitch, base 2 B off 4; 2 padded (1, 0): odd pitch
};

struct sdpa_hpg_ext_test : public ::testing::TestWithParam<sdpa_hpg_ext_params> {
    static constexpr int num_heads = 8;

    static std::string PrintToStringParamName(const testing::TestParamInfo<sdpa_hpg_ext_params>& info) {
        const auto& p = info.param;
        static const char* mask_names[] = {"nomask", "keymask", "fullmask", "constmask"};
        static const char* causal_names[] = {"nocausal", "causalTL", "causalLR"};
        std::string r = "head" + std::to_string(p.head_size) + "_q" + std::to_string(p.q_len) + "_kv" + std::to_string(p.kv_len) +
                        (p.bf16 ? "_bf16_" : "_f16_") + mask_names[p.mask] + "_" + causal_names[p.causal];
        if (p.sink)
            r += "_sink";
        r += p.dynamic ? "_dyn" : "_static";
        if (p.k_pad)
            r += "_kpad" + std::to_string(p.k_pad);
        return r;
    }

    static float round_to(float x, bool bf16) {
        return bf16 ? static_cast<float>(ov::bfloat16(x)) : static_cast<float>(ov::float16(x));
    }
    static void write(cldnn::memory::ptr m, const std::vector<float>& v, bool bf16) {
        if (bf16) {
            cldnn::mem_lock<ov::bfloat16, mem_lock_type::write> l(m, get_test_stream());
            ASSERT_GE(l.size(), v.size());
            for (size_t i = 0; i < v.size(); ++i)
                l[i] = ov::bfloat16(v[i]);
        } else {
            cldnn::mem_lock<ov::float16, mem_lock_type::write> l(m, get_test_stream());
            ASSERT_GE(l.size(), v.size());
            for (size_t i = 0; i < v.size(); ++i)
                l[i] = ov::float16(v[i]);
        }
    }
    static std::vector<float> read(cldnn::memory::ptr m, bool bf16) {
        std::vector<float> out;
        if (bf16) {
            cldnn::mem_lock<ov::bfloat16, mem_lock_type::read> l(m, get_test_stream());
            for (size_t i = 0; i < l.size(); ++i)
                out.push_back(static_cast<float>(l[i]));
        } else {
            cldnn::mem_lock<ov::float16, mem_lock_type::read> l(m, get_test_stream());
            for (size_t i = 0; i < l.size(); ++i)
                out.push_back(static_cast<float>(l[i]));
        }
        return out;
    }
};

TEST_P(sdpa_hpg_ext_test, plain_sdpa) {
    using T = sdpa_hpg_sharp_test;
    const auto p = GetParam();
    auto& engine = get_test_engine();
    if (tests::expected_dpas_backend(engine, false, static_cast<size_t>(p.head_size),
                                     ov::intel_gpu::ocl::PLAIN_F16_STATIC | ov::intel_gpu::ocl::PLAIN_EXT) != tests::dpas_backend::ocl)
        GTEST_SKIP() << "sdpa_ocl does not serve plain SDPA on this device (xe_hpg needs TEST_USE_SDPA_OCL_HPG=1 and TEST_USE_SDPA_OCL not 0)";
    ASSERT_TRUE(!p.sink || p.mask == 1 || p.mask == 2);
    ASSERT_TRUE(!(p.dynamic && p.k_pad)) << "the padded K cases are static";
    ASSERT_TRUE(p.mask != 2 || p.q_len > 1) << "a [.., 1, kv] mask is the per-key kind";

    const int H = num_heads, d = p.head_size, nq = p.q_len, nk = p.kv_len;
    const bool bf16 = p.bf16;
    const auto dt = bf16 ? data_types::bf16 : data_types::f16;
    const bool soft = p.mask == 1 || p.mask == 2;  // the mask decides, Q is noise
    const float key_gain = 32.0f;

    // Last key query i may see (causal), and the winner of (head, query): inside the visible keys in soft mode (the mask puts it
    // there), anywhere in sharp mode (a winner the causal mask hides then makes the kernel pick the best visible key, and a kernel
    // that forgot the mask pick the hidden one).
    const int lr_shift = std::max(0, nk - nq);
    auto last_visible = [&](int i) {
        if (p.causal == 0)
            return nk - 1;
        return std::min(nk - 1, p.causal == 2 ? i + lr_shift : i);
    };
    auto winner = [&](int h, int i) {
        const int span = soft ? last_visible(i) + 1 : nk;
        return static_cast<int>((17u * static_cast<uint32_t>(i) + 3u + 5u * static_cast<uint32_t>(h)) % static_cast<uint32_t>(span));
    };

    std::vector<float> q(static_cast<size_t>(H) * nq * d), k(static_cast<size_t>(H) * nk * d), v(static_cast<size_t>(H) * nk * d);
    for (int h = 0; h < H; ++h) {
        for (int j = 0; j < nk; ++j)
            for (int c = 0; c < d; ++c) {
                const size_t at = (static_cast<size_t>(h) * nk + j) * d + c;
                k[at] = T::grid_value(1, h, j, c);
                v[at] = T::grid_value(2, h, j, c) * 2.0f;
            }
        for (int i = 0; i < nq; ++i)
            for (int c = 0; c < d; ++c) {
                const size_t at = (static_cast<size_t>(h) * nq + i) * d + c;
                q[at] = soft ? T::grid_value(3, h, i, c) : key_gain * T::grid_value(1, h, winner(h, i), c);
            }
    }

    // Mask (stored values, in logit units). Per key: one winner per head. Full: one winner per (head, query), some other keys -inf.
    std::vector<float> mask;
    if (p.mask == 1) {
        mask.resize(static_cast<size_t>(H) * nk);
        for (int h = 0; h < H; ++h) {
            const int w = (7 * h + 3) % nk;
            for (int j = 0; j < nk; ++j)
                mask[static_cast<size_t>(h) * nk + j] = round_to(-2.5f * static_cast<float>((j - w + nk) % nk), bf16);
        }
    } else if (p.mask == 2) {
        mask.resize(static_cast<size_t>(H) * nq * nk);
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < nq; ++i) {
                const int w = winner(h, i);
                for (int j = 0; j < nk; ++j) {
                    float m = -2.5f * static_cast<float>(std::abs(j - w));
                    if (j != w && (3 * i + j + h) % 11 == 7)
                        m = -INFINITY;
                    mask[(static_cast<size_t>(h) * nq + i) * nk + j] = round_to(m, bf16);
                }
            }
    }
    const float const_mask_val = 0.5f;
    std::vector<float> sink_vals(H), scale_val{round_to(1.0f / std::sqrt(static_cast<float>(d)), bf16)};
    for (int h = 0; h < H; ++h)
        sink_vals[h] = round_to(0.5f * static_cast<float>(h - 3), bf16);

    // Layouts. K may be a padded view (static only): physical row = d + lower + upper elements, the data starts `lower` in.
    static const int kpad_lo[] = {0, 1, 1, 2, 0, 3}, kpad_up[] = {0, 1, 0, 0, 1, 1};
    const int k_lo = kpad_lo[p.k_pad], k_up = kpad_up[p.k_pad];
    const layout q_static({1, H, nq, d}, dt, format::bfyx);
    layout k_static({1, H, nk, d}, dt, format::bfyx);
    k_static.data_padding._lower_size[3] = k_lo;
    k_static.data_padding._upper_size[3] = k_up;
    const layout v_static({1, H, nk, d}, dt, format::bfyx);
    const layout q_lay = p.dynamic ? layout({1, H, -1, d}, dt, format::bfyx) : q_static;
    const layout kv_lay = p.dynamic ? layout({1, H, -1, d}, dt, format::bfyx) : k_static;
    const layout v_lay = p.dynamic ? layout({1, H, -1, d}, dt, format::bfyx) : v_static;
    const layout mask_static = p.mask == 1 ? layout({1, H, 1, nk}, dt, format::bfyx) : layout({1, H, nq, nk}, dt, format::bfyx);
    const layout mask_lay = !p.dynamic ? mask_static : p.mask == 1 ? layout({1, H, 1, -1}, dt, format::bfyx) : layout({1, H, -1, -1}, dt, format::bfyx);

    auto q_mem = engine.allocate_memory(q_static);
    auto k_mem = engine.allocate_memory(k_static);
    auto v_mem = engine.allocate_memory(v_static);
    write(q_mem, q, bf16);
    write(v_mem, v, bf16);
    {
        const size_t row = static_cast<size_t>(d + k_lo + k_up);
        std::vector<float> k_phys(static_cast<size_t>(H) * nk * row, 0.0f);
        for (size_t r = 0; r < static_cast<size_t>(H) * nk; ++r)
            for (int c = 0; c < d; ++c)
                k_phys[r * row + k_lo + c] = k[r * d + c];
        write(k_mem, k_phys, bf16);
    }

    topology topo;
    topo.add(input_layout("q", q_lay));
    topo.add(input_layout("k", kv_lay));
    topo.add(input_layout("v", v_lay));
    std::vector<input_info> inputs{input_info("q"), input_info("k"), input_info("v")};
    if (p.mask == 1 || p.mask == 2) {
        topo.add(input_layout("mask", mask_lay));
        inputs.push_back(input_info("mask"));
    }
    if (p.sink) {
        topo.add(input_layout("scale", layout({1, 1, 1, 1}, dt, format::bfyx)));
        topo.add(input_layout("sink", layout({1, H, 1, 1}, dt, format::bfyx)));
        inputs.push_back(input_info("scale"));
        inputs.push_back(input_info("sink"));
    }
    auto prim = scaled_dot_product_attention("sdpa", inputs, p.causal != 0, -1, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {}, false,
                                             p.causal == 2);
    if (p.mask == 3)
        prim.attn_mask_val = const_mask_val;
    topo.add(prim);
    topo.add(reorder("result", input_info("sdpa"), format::bfyx, dt));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
    net->set_input_data("q", q_mem);
    net->set_input_data("k", k_mem);
    net->set_input_data("v", v_mem);
    if (p.mask == 1 || p.mask == 2) {
        auto m = engine.allocate_memory(mask_static);
        write(m, mask, bf16);
        net->set_input_data("mask", m);
    }
    if (p.sink) {
        auto sc = engine.allocate_memory(layout({1, 1, 1, 1}, dt, format::bfyx));
        auto sk = engine.allocate_memory(layout({1, H, 1, 1}, dt, format::bfyx));
        write(sc, scale_val, bf16);
        write(sk, sink_vals, bf16);
        net->set_input_data("scale", sc);
        net->set_input_data("sink", sk);
    }
    auto output = net->execute().at("result").get_memory();

    // The kernel that ran: sdpa_ocl, the prefill stage for q > 1 and the single-token stage for q == 1 (a dynamic op builds both).
    auto sdpa_inst = net->get_primitive("sdpa");
    ASSERT_NE(sdpa_inst, nullptr);
    ASSERT_NE(sdpa_inst->get_impl(), nullptr);
    const auto entries = sdpa_inst->get_impl()->get_kernels_dump_info(*sdpa_inst->get_impl_params()).get_entries();
    ASSERT_NE(entries.find("sdpa_ocl"), std::string::npos) << "dispatched: " << entries;
    EXPECT_EQ(entries.find("sdpa_micro"), std::string::npos) << "dispatched: " << entries;
    if (!p.dynamic) {
        ASSERT_NE(entries.find(nq > 1 ? "sdpa_ocl_prefill" : "sdpa_ocl_mixed"), std::string::npos) << "dispatched: " << entries;
    }

    // CPU reference in double over the stored (rounded) inputs: softmax over [logits..., sink], the sink column dropped.
    const auto out = read(output, bf16);
    ASSERT_EQ(out.size(), static_cast<size_t>(H) * nq * d);
    const double scale = p.sink ? static_cast<double>(scale_val[0]) : 1.0 / std::sqrt(static_cast<double>(d));
    double max_abs_err = 0.0;
    double err_by_qblock[2] = {0.0, 0.0};  // max error of the even / odd 8-query blocks (the two DPAS query blocks of a subgroup)
    std::vector<double> logit(nk), acc(d);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < nq; ++i) {
            const int last = last_visible(i);
            double max_logit = p.sink ? static_cast<double>(sink_vals[h]) : -INFINITY;
            for (int j = 0; j < nk; ++j) {
                logit[j] = -INFINITY;
                if (j > last)
                    continue;
                double dot = 0.0;
                for (int c = 0; c < d; ++c)
                    dot += static_cast<double>(q[(static_cast<size_t>(h) * nq + i) * d + c]) * static_cast<double>(k[(static_cast<size_t>(h) * nk + j) * d + c]);
                double l = dot * scale;
                if (p.mask == 1)
                    l += mask[static_cast<size_t>(h) * nk + j];
                else if (p.mask == 2)
                    l += mask[(static_cast<size_t>(h) * nq + i) * nk + j];
                else if (p.mask == 3)
                    l += const_mask_val;
                logit[j] = l;
                max_logit = std::max(max_logit, l);
            }
            double sum = p.sink ? std::exp(static_cast<double>(sink_vals[h]) - max_logit) : 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < nk; ++j) {
                if (logit[j] == -INFINITY)
                    continue;
                const double w = std::exp(logit[j] - max_logit);
                sum += w;
                for (int c = 0; c < d; ++c)
                    acc[c] += w * static_cast<double>(v[(static_cast<size_t>(h) * nk + j) * d + c]);
            }
            for (int c = 0; c < d; ++c) {
                const float got = out[(static_cast<size_t>(h) * nq + i) * d + c];
                ASSERT_FALSE(std::isnan(got)) << "NaN at head " << h << " query " << i << " col " << c;
                const double err = std::abs(static_cast<double>(got) - acc[c] / sum);
                max_abs_err = std::max(max_abs_err, err);
                err_by_qblock[(i / 8) & 1] = std::max(err_by_qblock[(i / 8) & 1], err);
            }
        }
    }
    RecordProperty("max_abs_err", std::to_string(max_abs_err));
    if (max_abs_err > (bf16 ? 2e-2 : 1e-2))
        std::cout << "DIAG max err by query block (i / 8 even, odd): " << err_by_qblock[0] << " " << err_by_qblock[1] << std::endl;
    EXPECT_LE(max_abs_err, bf16 ? 2e-2 : 1e-2) << sdpa_hpg_ext_test::PrintToStringParamName(testing::TestParamInfo<sdpa_hpg_ext_params>(p, 0))
                                               << " max_abs_err " << max_abs_err;
}

// One axis at a time away from the base (f16, static, head 64, 100 x 100, nothing extra), plus the risky combinations. 100 is not a
// multiple of any tile; 17 / 33 straddle the 8 / 16 / 32 key tiles.
// {head, q, kv, bf16, mask, causal, sink, dynamic, k_pad}
INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_hpg_ext,
    sdpa_hpg_ext_test,
    ::testing::Values(
        // bf16 (sharp)
        sdpa_hpg_ext_params{64, 100, 100, true, 0, 0, false, false, 0}, sdpa_hpg_ext_params{128, 64, 64, true, 0, 0, false, false, 0},
        sdpa_hpg_ext_params{72, 100, 100, true, 0, 0, false, false, 0}, sdpa_hpg_ext_params{256, 100, 100, true, 0, 0, false, false, 0},
        // per-key mask (kind 1), tile-boundary lengths
        sdpa_hpg_ext_params{64, 17, 17, false, 1, 0, false, false, 0}, sdpa_hpg_ext_params{64, 33, 33, false, 1, 0, false, false, 0},
        sdpa_hpg_ext_params{64, 100, 100, false, 1, 0, false, false, 0}, sdpa_hpg_ext_params{128, 256, 256, false, 1, 0, false, false, 0},
        sdpa_hpg_ext_params{64, 100, 100, true, 1, 0, false, false, 0},
        // full 2D mask (kind 2)
        sdpa_hpg_ext_params{64, 17, 17, false, 2, 0, false, false, 0}, sdpa_hpg_ext_params{64, 33, 33, false, 2, 0, false, false, 0},
        sdpa_hpg_ext_params{64, 100, 100, false, 2, 0, false, false, 0}, sdpa_hpg_ext_params{128, 256, 256, false, 2, 0, false, false, 0},
        sdpa_hpg_ext_params{72, 100, 100, false, 2, 0, false, false, 0}, sdpa_hpg_ext_params{64, 100, 100, true, 2, 0, false, false, 0},
        sdpa_hpg_ext_params{64, 16, 64, false, 2, 0, false, false, 0},
        // const scalar mask (does it run; softmax is shift invariant)
        sdpa_hpg_ext_params{64, 100, 100, false, 3, 0, false, false, 0},
        // causal, both alignments, q != kv
        sdpa_hpg_ext_params{64, 100, 100, false, 0, 1, false, false, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 0, 2, false, false, 0},
        sdpa_hpg_ext_params{64, 16, 64, false, 0, 2, false, false, 0}, sdpa_hpg_ext_params{64, 16, 64, false, 0, 1, false, false, 0},
        sdpa_hpg_ext_params{128, 256, 256, false, 0, 1, false, false, 0}, sdpa_hpg_ext_params{64, 100, 100, true, 0, 1, false, false, 0},
        // sink (needs the mask to decide)
        sdpa_hpg_ext_params{64, 100, 100, false, 2, 0, true, false, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 1, 0, true, false, 0},
        sdpa_hpg_ext_params{128, 256, 256, false, 2, 0, true, false, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 2, 1, true, false, 0},
        // dynamic shape
        sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, true, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 2, 0, false, true, 0},
        sdpa_hpg_ext_params{64, 100, 100, false, 1, 0, false, true, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 0, 1, false, true, 0},
        sdpa_hpg_ext_params{64, 100, 100, true, 2, 1, false, true, 0}, sdpa_hpg_ext_params{64, 100, 100, false, 2, 1, true, true, 0},
        // the risky product: full mask x causal x bf16 x dynamic x sink
        sdpa_hpg_ext_params{128, 100, 100, true, 2, 2, true, true, 0}, sdpa_hpg_ext_params{128, 100, 100, false, 2, 2, true, true, 0},
        sdpa_hpg_ext_params{64, 100, 100, true, 2, 0, true, false, 0}, sdpa_hpg_ext_params{64, 100, 100, true, 1, 0, true, false, 0},
        sdpa_hpg_ext_params{128, 100, 100, true, 2, 2, false, true, 0},
        // single query (the sdpa_ocl_mixed stage)
        sdpa_hpg_ext_params{64, 1, 100, false, 0, 0, false, false, 0}, sdpa_hpg_ext_params{64, 1, 1000, false, 1, 0, false, false, 0},
        sdpa_hpg_ext_params{64, 1, 100, true, 1, 0, false, false, 0}, sdpa_hpg_ext_params{128, 1, 100, false, 0, 2, false, false, 0},
        sdpa_hpg_ext_params{64, 1, 100, false, 1, 0, true, false, 0}, sdpa_hpg_ext_params{64, 1, 100, false, 1, 0, false, true, 0},
        sdpa_hpg_ext_params{64, 1, 100, false, 0, 0, false, true, 0},
        // K views that break the dword read (even pitch, base 2 B off; odd pitch): the runtime fallback
        sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, false, 1}, sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, false, 2},
        sdpa_hpg_ext_params{72, 100, 100, true, 0, 0, false, false, 1},
        sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, false, 3}, sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, false, 4},
        sdpa_hpg_ext_params{64, 100, 100, false, 0, 0, false, false, 5}
    ),
    sdpa_hpg_ext_test::PrintToStringParamName
);
#endif

enum class sdpa_ref_accuracy_case { uniform_16, uniform_32, uniform_64, nonuniform, nonuniform_33, nonuniform_100, mask, causal };

class sdpa_ref_accuracy_test
    : public ::testing::TestWithParam<std::tuple<data_types, sdpa_ref_accuracy_case>> {
protected:
    template <typename T>
    void check_accuracy(data_types dt, sdpa_ref_accuracy_case test_case) {
        auto& engine = get_test_engine();
        RecordProperty("device", engine.get_device_info().dev_name);
        const bool uniform = test_case == sdpa_ref_accuracy_case::uniform_16 ||
                             test_case == sdpa_ref_accuracy_case::uniform_32 ||
                             test_case == sdpa_ref_accuracy_case::uniform_64;
        const bool use_mask = test_case == sdpa_ref_accuracy_case::mask;
        const bool causal = test_case == sdpa_ref_accuracy_case::causal;
        const size_t seq_q = use_mask || causal ? 4 : 1;
        const size_t seq_kv = test_case == sdpa_ref_accuracy_case::uniform_16 ? 16 :
                             test_case == sdpa_ref_accuracy_case::uniform_64 ? 64 :
                             test_case == sdpa_ref_accuracy_case::nonuniform_33 ? 33 :
                             test_case == sdpa_ref_accuracy_case::nonuniform_100 ? 100 : 32;
        const ov::Shape q_shape{2, 4, seq_q, 64};
        const ov::Shape kv_shape{2, 4, seq_kv, 64};
        std::array<std::vector<T>, 3> input_data;
        std::array<std::vector<float>, 3> reference_data;
        std::array<memory::ptr, 3> inputs;
        const std::array<std::string, 3> names{"q", "k", "v"};
        topology topo;
        for (size_t input = 0; input < inputs.size(); ++input) {
            const auto& shape = input == 0 ? q_shape : kv_shape;
            input_data[input].resize(ov::shape_size(shape));
            reference_data[input].resize(input_data[input].size());
            for (size_t b = 0; b < shape[0]; ++b)
                for (size_t h = 0; h < shape[1]; ++h)
                    for (size_t s = 0; s < shape[2]; ++s)
                        for (size_t e = 0; e < shape[3]; ++e) {
                            const auto index = ((b * shape[1] + h) * shape[2] + s) * shape[3] + e;
                            float value = std::sin(float((b + 1) * 73 + h * 29 + s * 17 + e * 11 + input * 47) * 0.07f);
                            if (input == 2)
                                value = value * 0.5f + float(b) * 0.3f;
                            if (uniform)
                                value = input == 2 ? 0.25f + float(b) * 0.5f : 0.f;
                            input_data[input][index] = T(value);
                            reference_data[input][index] = static_cast<float>(input_data[input][index]);
                        }
            inputs[input] = engine.allocate_memory(layout(shape, dt, format::bfyx));
            set_values(inputs[input], input_data[input]);
            auto parameter_shape = ov::PartialShape(shape);
            // A dynamic V head size selects SDPARef without relying on kernel-name forcing.
            if (input == 2)
                parameter_shape[3] = ov::Dimension::dynamic();
            topo.add(input_layout(names[input], layout(parameter_shape, dt, format::bfyx)));
        }

        const ov::Shape mask_shape{1, 1, seq_q, seq_kv};
        std::vector<T> mask_data;
        std::vector<float> reference_mask;
        std::vector<input_info> sdpa_inputs{input_info("q"), input_info("k"), input_info("v")};
        memory::ptr mask_mem;
        if (use_mask) {
            mask_data.resize(ov::shape_size(mask_shape));
            reference_mask.resize(mask_data.size());
            for (size_t i = 0; i < mask_data.size(); ++i) {
                mask_data[i] = T(i % seq_kv == seq_kv - 1 ? -std::numeric_limits<float>::infinity() :
                                                              -float(i % 5) * 0.125f);
                reference_mask[i] = static_cast<float>(mask_data[i]);
            }
            mask_mem = engine.allocate_memory(layout(mask_shape, dt, format::bfyx));
            set_values(mask_mem, mask_data);
            topo.add(input_layout("mask", mask_mem->get_layout()));
            sdpa_inputs.emplace_back("mask");
        }
        const std::vector<int64_t> order{0, 1, 2, 3};
        auto sdpa = scaled_dot_product_attention("sdpa", sdpa_inputs, causal, -1, order, order, order, order, {}, false);
        const float scale = 0.125f;
        topo.add(sdpa);
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto net = get_network(engine, topo, config, get_test_stream_ptr(), false);
        for (size_t input = 0; input < inputs.size(); ++input)
            net->set_input_data(names[input], inputs[input]);
        if (use_mask)
            net->set_input_data("mask", mask_mem);
        auto output = net->execute().at("sdpa").get_memory();
        ASSERT_EQ(net->get_primitive("sdpa")->get_impl()->m_manager->get_type_info(),
                  ov::intel_gpu::ocl::SDPARef::get_type_info_static());
        ASSERT_EQ(output->get_layout().get_shape(), q_shape);
        ASSERT_EQ(output->get_layout().data_type, dt);

        std::vector<float> expected(ov::shape_size(q_shape));
        ov::reference::scaled_dot_product_attention<float, float>(reference_data[0].data(),
                                                                 reference_data[1].data(),
                                                                 reference_data[2].data(),
                                                                 use_mask ? reference_mask.data() : nullptr,
                                                                 &scale,
                                                                 nullptr,
                                                                 expected.data(),
                                                                 causal,
                                                                 q_shape,
                                                                 kv_shape,
                                                                 kv_shape,
                                                                 mask_shape,
                                                                 {},
                                                                 q_shape);
        mem_lock<T, mem_lock_type::read> output_data(output, get_test_stream());
        ASSERT_EQ(output_data.size(), expected.size());
        const float tolerance = dt == data_types::f32 ? 1e-5f : 5e-3f;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float actual = static_cast<float>(output_data[i]);
            ASSERT_TRUE(std::isfinite(actual)) << "index=" << i;
            if (uniform)
                ASSERT_NEAR(actual, 0.25f + float(i / (4 * seq_q * 64)) * 0.5f, 1e-3f) << "index=" << i;
            else
                ASSERT_NEAR(actual, expected[i], tolerance) << "index=" << i;
        }
    }
};

TEST_P(sdpa_ref_accuracy_test, matches_independent_reference) {
    const auto [dt, test_case] = GetParam();
    if (dt == data_types::f16)
        check_accuracy<ov::float16>(dt, test_case);
    else if (dt == data_types::bf16)
        check_accuracy<ov::bfloat16>(dt, test_case);
    else
        check_accuracy<float>(dt, test_case);
}

INSTANTIATE_TEST_SUITE_P(
    sdpa_ref,
    sdpa_ref_accuracy_test,
    ::testing::Combine(::testing::Values(data_types::f16, data_types::f32, data_types::bf16),
                       ::testing::Values(sdpa_ref_accuracy_case::uniform_16,
                                         sdpa_ref_accuracy_case::uniform_32,
                                         sdpa_ref_accuracy_case::uniform_64,
                                         sdpa_ref_accuracy_case::nonuniform,
                                         sdpa_ref_accuracy_case::nonuniform_33,
                                         sdpa_ref_accuracy_case::nonuniform_100,
                                         sdpa_ref_accuracy_case::mask,
                                         sdpa_ref_accuracy_case::causal)),
    ([](const ::testing::TestParamInfo<sdpa_ref_accuracy_test::ParamType>& info) {
        const auto [dt, test_case] = info.param;
        const std::array<std::string, 8> names{"Uniform16", "Uniform32", "Uniform64", "Nonuniform32",
                                             "Nonuniform33", "Nonuniform100", "Mask", "Causal"};
        return (dt == data_types::f16 ? "FP16" : dt == data_types::f32 ? "FP32" : "BF16") +
               names[static_cast<size_t>(test_case)];
    }));

TEST(sdpa_gpu_custom, ref_fp16_accuracy_reused_dynamic_batch) {
    auto& engine = get_test_engine();
    ::testing::Test::RecordProperty("device", engine.get_device_info().dev_name);
    topology topo;
    topo.add(input_layout("q", layout(ov::PartialShape{-1, 4, 1, 64}, data_types::f16, format::bfyx)));
    topo.add(input_layout("k", layout(ov::PartialShape{-1, 4, 32, 64}, data_types::f16, format::bfyx)));
    topo.add(input_layout("v", layout(ov::PartialShape{-1, 4, 32, -1}, data_types::f16, format::bfyx)));
    const std::vector<int64_t> order{0, 1, 2, 3};
    topo.add(scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v")},
                                          false, -1, order, order, order, order, {}, false));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto net = get_network(engine, topo, config, get_test_stream_ptr(), false);
    for (size_t batch : {1, 2, 1}) {
        SCOPED_TRACE(batch);
        auto q = engine.allocate_memory(layout(ov::Shape{batch, 4, 1, 64}, data_types::f16, format::bfyx));
        auto k = engine.allocate_memory(layout(ov::Shape{batch, 4, 32, 64}, data_types::f16, format::bfyx));
        auto v = engine.allocate_memory(k->get_layout());
        set_values(q, std::vector<ov::float16>(q->get_layout().count(), ov::float16(0.f)));
        set_values(k, std::vector<ov::float16>(k->get_layout().count(), ov::float16(0.f)));
        std::vector<ov::float16> values(v->get_layout().count());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = ov::float16(0.25f + float(i / (4 * 32 * 64)) * 0.5f);
        set_values(v, values);
        net->set_input_data("q", q);
        net->set_input_data("k", k);
        net->set_input_data("v", v);
        auto output = net->execute().at("sdpa").get_memory();
        ASSERT_EQ(net->get_primitive("sdpa")->get_impl()->m_manager->get_type_info(),
                  ov::intel_gpu::ocl::SDPARef::get_type_info_static());
        ASSERT_EQ(output->get_layout().get_shape(), (ov::Shape{batch, 4, 1, 64}));
        mem_lock<ov::float16, mem_lock_type::read> output_data(output, get_test_stream());
        for (size_t i = 0; i < output_data.size(); ++i)
            ASSERT_NEAR(static_cast<float>(output_data[i]), 0.25f + float(i / (4 * 64)) * 0.5f, 1e-3f) << "index=" << i;
    }
}

TEST(sdpa_gpu_custom, dynamic_mismatched_v_head_size) {
    auto& engine = get_test_engine();

    const ov::PartialShape qk_shape{-1, 1, -1, 384};
    const ov::PartialShape v_shape{-1, 1, -1, -1};
    const ov::Shape qk_static_shape{1, 1, 16, 384};
    const ov::Shape v_static_shape{1, 1, 16, 256};

    const layout q_layout(qk_shape, data_types::f16, format::bfyx);
    const layout k_layout(qk_shape, data_types::f16, format::bfyx);
    const layout v_layout(v_shape, data_types::f16, format::bfyx);
    const layout qk_static_layout(qk_static_shape, data_types::f16, format::bfyx);
    const layout v_static_layout(v_static_shape, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(qk_static_layout);
    auto k_mem = engine.allocate_memory(qk_static_layout);
    auto v_mem = engine.allocate_memory(v_static_layout);

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto fill_random = [&](const memory::ptr& mem) {
        auto data = rg.generate_random_1d<ov::float16>(mem->get_layout().count(), -1.0f, 1.0f);
        set_values(mem, data);
    };
    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);

    topology topology;
    topology.add(input_layout("q", q_layout));
    topology.add(input_layout("k", k_layout));
    topology.add(input_layout("v", v_layout));
    topology.add(scaled_dot_product_attention("sdpa",
                                              {input_info("q"), input_info("k"), input_info("v")},
                                              false,
                                              -1,
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {},
                                              false));
    topology.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    auto run_network = [&](bool force_ref) {
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (force_ref) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                {"sdpa", {format::type::bfyx, "sdpa_ref"}}
            }));
        }

        auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        auto output = network->execute().at("result").get_memory();
        return std::make_pair(network, output);
    };

    auto [network, output] = run_network(false);
    auto [ref_network, ref_output] = run_network(true);

    ASSERT_NE(network->get_primitive_info("sdpa").find("sdpa_ref"), std::string::npos);
    ASSERT_EQ(output->get_layout().get_shape(), v_static_shape);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_data(output, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_output_data(ref_output, get_test_stream());
    ASSERT_EQ(output_data.size(), ref_output_data.size());
    for (size_t i = 0; i < output_data.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_data[i]), static_cast<float>(ref_output_data[i]), 1e-3f)
            << "Mismatch at index " << i;
    }
}

TEST(sdpa_gpu_custom, different_rank_orders_dynamic_v_head_size_rejects_opt) {
    auto& engine = get_test_engine();

    auto q_prim = std::make_shared<input_layout>("q", layout(ov::PartialShape{-1, -1, 384}, data_types::f16, format::bfyx));
    auto k_prim = std::make_shared<input_layout>("k", layout(ov::PartialShape{-1, -1, 384}, data_types::f16, format::bfyx));
    auto v_prim = std::make_shared<input_layout>("v", layout(ov::PartialShape{-1, 1, 16, -1}, data_types::f16, format::bfyx));
    auto sdpa_prim = std::make_shared<scaled_dot_product_attention>("sdpa",
                                                                   std::vector{input_info("q"), input_info("k"), input_info("v")},
                                                                   false,
                                                                   -1,
                                                                   std::vector<int64_t>{0, 1, 2},
                                                                   std::vector<int64_t>{0, 1, 2},
                                                                   std::vector<int64_t>{0, 1, 2, 3},
                                                                   std::vector<int64_t>{0, 1, 2, 3},
                                                                   scaled_dot_product_attention::QuantizationAttributes{},
                                                                   false);

    program prog(engine);
    auto& q_node = prog.get_or_create(q_prim);
    auto& k_node = prog.get_or_create(k_prim);
    auto& v_node = prog.get_or_create(v_prim);
    auto& sdpa_node = prog.get_or_create(sdpa_prim);
    program_wrapper::add_connection(prog, q_node, sdpa_node);
    program_wrapper::add_connection(prog, k_node, sdpa_node);
    program_wrapper::add_connection(prog, v_node, sdpa_node);
    sdpa_node.recalc_output_layout();

    ov::intel_gpu::ocl::SDPAOpt sdpa_opt(shape_types::dynamic_shape);
    EXPECT_FALSE(sdpa_opt.validate_impl(sdpa_node));
}

TEST(sdpa_gpu_custom, static_zero_dimension_throws) {
    auto& engine = get_test_engine();

    const std::vector<std::array<ov::Shape, 3>> input_shapes = {
        {{{1, 0, 16, 32}, {1, 0, 16, 32}, {1, 0, 16, 32}}},
        {{{1, 1, 16, 32}, {1, 1, 16, 32}, {1, 1, 16, 0}}},
    };

    for (const auto& shapes : input_shapes) {
        topology topology;
        topology.add(input_layout("q", layout(shapes[0], data_types::f16, format::bfyx)));
        topology.add(input_layout("k", layout(shapes[1], data_types::f16, format::bfyx)));
        topology.add(input_layout("v", layout(shapes[2], data_types::f16, format::bfyx)));
        topology.add(scaled_dot_product_attention("sdpa",
                                                  {input_info("q"), input_info("k"), input_info("v")},
                                                  false,
                                                  -1,
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {},
                                                  false));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
            {"sdpa", {format::type::bfyx, "sdpa_ref"}}
        }));

        EXPECT_ANY_THROW(get_network(engine, topology, config, get_test_stream_ptr(), false));
    }
}

TEST(sdpa_gpu_custom, dynamic_zero_head_size_throws) {
    auto& engine = get_test_engine();

    const ov::PartialShape dynamic_shape{-1, 1, -1, -1};
    const layout dynamic_layout(dynamic_shape, data_types::f16, format::bfyx);
    auto q_mem = engine.allocate_memory(layout({1, 1, 16, 0}, data_types::f16, format::bfyx));
    auto k_mem = engine.allocate_memory(layout({1, 1, 16, 0}, data_types::f16, format::bfyx));
    auto v_mem = engine.allocate_memory(layout({1, 1, 16, 32}, data_types::f16, format::bfyx));

    topology topology;
    topology.add(input_layout("q", dynamic_layout));
    topology.add(input_layout("k", dynamic_layout));
    topology.add(input_layout("v", dynamic_layout));
    topology.add(scaled_dot_product_attention("sdpa",
                                              {input_info("q"), input_info("k"), input_info("v")},
                                              false,
                                              -1,
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {},
                                              false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"sdpa", {format::type::bfyx, "sdpa_ref"}}
    }));

    auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->set_input_data("q", q_mem);
    network->set_input_data("k", k_mem);
    network->set_input_data("v", v_mem);
    try {
        network->execute();
        FAIL() << "Expected runtime SDPA dimension validation to throw";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("invalid non-positive q_head_size for runtime dispatch"), std::string::npos)
            << "Unexpected exception message: " << e.what();
    }
}

TEST(sdpa_gpu_custom, single_token_cond_attn_mask_clamp) {
    tests::random_generator rg; rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (engine.get_device_info().supports_immad) {
        return;
    }

    const int head_size = 32;
    const int num_heads = 1;
    const int seq_length_q = 1;
    const int seq_length_kv = 448;
    const int batch = 1;

    layout input0_layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
    layout input1_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input2_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input3_layout({batch, num_heads, 1, seq_length_kv}, data_types::f16, format::bfyx);

    auto input0 = engine.allocate_memory(input0_layout);
    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);
    auto input3 = engine.allocate_memory(input3_layout);


    auto fill_random = [&](memory::ptr mem) {
        auto shp = mem->get_layout().get_shape();
        size_t sz = ov::shape_size(shp);
        auto data = rg.generate_random_1d<ov::float16>(sz, -1.0f, 1.0f);
        set_values(mem, data);
    };
    fill_random(input0);
    fill_random(input1);
    fill_random(input2);

    // attention mask with first position 0, all remaining positions -inf
    {
        size_t elems = batch * num_heads * 1 * seq_length_kv;
        std::vector<ov::float16> mask(elems);
        for (int kv = 0; kv < seq_length_kv; ++kv) {
            mask[kv] = kv == 0 ? ov::float16(0.0f) : ov::float16(-std::numeric_limits<float>::infinity());
        }
        set_values(input3, mask);
    }

    topology topology;
    topology.add(input_layout("input0", input0_layout));
    topology.add(input_layout("input1", input1_layout));
    topology.add(input_layout("input2", input2_layout));
    topology.add(input_layout("input3", input3_layout));
    topology.add(scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
                                              false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
    topology.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"sdpa", {format::type::bfyx, "sdpa_opt"}}
    }));
    auto network = get_network(engine, topology, cfg, get_test_stream_ptr(), false);
    network->set_input_data("input0", input0);
    network->set_input_data("input1", input1);
    network->set_input_data("input2", input2);
    network->set_input_data("input3", input3);
    auto output = network->execute().at("result").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(output_ptr[i])));
    }

    // With only first KV valid, output should approximate value vector at KV index 0.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(input2, get_test_stream());
    for (int hs = 0; hs < head_size; ++hs) {
        ASSERT_NEAR(static_cast<float>(ref_ptr[hs]), static_cast<float>(output_ptr[hs]), 1e-2f);
    }
}

TEST(sdpa_gpu_custom, scalar_placeholder_mask_matches_scale_only) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch = 1;
    const int seq_length_q = 4;
    const int seq_length_kv = 6;
    const int num_heads = 2;
    const int head_size = 32;
    const float scale_val = 0.35f;

    const layout q_layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout k_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout v_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout scalar_mask_layout{ov::PartialShape{}, data_types::f16, format::bfyx};

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto v_mem = engine.allocate_memory(v_layout);
    auto scalar_mask_mem = engine.allocate_memory(scalar_mask_layout);

    auto fill_random = [&](const memory::ptr& mem) {
        const auto shape = mem->get_layout().get_shape();
        const size_t elements_num = ov::shape_size(shape);
        auto data = rg.generate_random_1d<ov::float16>(elements_num, -1.0f, 1.0f);
        set_values(mem, data);
    };

    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);
    set_values(scalar_mask_mem, {ov::float16(1.0f)});

    auto run_sdpa = [&](bool use_placeholder_mask) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        std::vector<input_info> inputs = {input_info("q"), input_info("k"), input_info("v")};
        if (use_placeholder_mask) {
            topo.add(input_layout("mask", scalar_mask_layout));
            inputs.push_back(input_info("mask"));
        }

        auto sdpa_prim = scaled_dot_product_attention("sdpa",
                                                      inputs,
                                                      false,
                                                      -1,
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 1, 2, 3},
                                                      {},
                                                      false);
        sdpa_prim.scale_val = scale_val;

        topo.add(sdpa_prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto network = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        if (use_placeholder_mask) {
            network->set_input_data("mask", scalar_mask_mem);
        }

        return network->execute().at("result").get_memory();
    };

    auto output_without_mask = run_sdpa(false);
    auto output_with_placeholder_mask = run_sdpa(true);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> without_mask_ptr(output_without_mask, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> with_placeholder_mask_ptr(output_with_placeholder_mask, get_test_stream());

    ASSERT_EQ(without_mask_ptr.size(), with_placeholder_mask_ptr.size());
    for (size_t i = 0; i < without_mask_ptr.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(without_mask_ptr[i]), static_cast<float>(with_placeholder_mask_ptr[i]), 1e-3f)
            << "Mismatch at index " << i
            << ", Expected : " << static_cast<float>(without_mask_ptr[i])
            << " actual : " << static_cast<float>(with_placeholder_mask_ptr[i])
            << std::endl;
    }
}

struct sdpa_ref_scratch_test : public ::testing::TestWithParam<std::tuple<data_types, int, bool>> {};

TEST_P(sdpa_ref_scratch_test, native_q_broadcast) {
    const auto dt = std::get<0>(GetParam());
    const auto order_kind = std::get<1>(GetParam());
    const auto dynamic_batch = std::get<2>(GetParam());
    auto& engine = get_test_engine();

    auto dims = [&](int64_t batch, int64_t sequence, int64_t head_size) {
        switch (order_kind) {
        case 1:
            return ov::PartialShape{batch, sequence, 4, head_size};
        case 2:
            return ov::PartialShape{sequence, batch, 4, head_size};
        default:
            return ov::PartialShape{batch, 4, sequence, head_size};
        }
    };
    const std::vector<std::vector<int64_t>> orders = {{0, 1, 2, 3}, {0, 2, 1, 3}, {1, 2, 0, 3}};
    const auto& order = orders.at(order_kind);
    const int64_t kv_batch = dynamic_batch ? -1 : 2;
    topology topo;
    topo.add(input_layout("q", layout(dims(1, 2, 64), dt, format::bfyx)));
    topo.add(input_layout("k", layout(dims(kv_batch, 16, 64), dt, format::bfyx)));
    // A dynamic V head size naturally selects SDPARef, without implementation forcing.
    topo.add(input_layout("v", layout(dims(kv_batch, 16, -1), dt, format::bfyx)));
    topo.add(
        scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v")}, false, -1, order, order, order, {0, 1, 2, 3}, {}, false));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto net = get_network(engine, topo, config, get_test_stream_ptr(), false);
    auto& stream = net->get_stream();
    const auto sdpa = net->get_primitive("sdpa");
    ASSERT_EQ(sdpa->get_impl()->m_manager->get_type_info(), ov::intel_gpu::ocl::SDPARef::get_type_info_static());
    const auto initial_descriptors = sdpa->get_impl()->get_internal_buffer_descs(*sdpa->get_impl_params());
    ASSERT_EQ(initial_descriptors.size(), 1u);
    ASSERT_EQ(initial_descriptors[0].m_layout.count(), 1u);

    const std::vector<int64_t> batches = dynamic_batch ? std::vector<int64_t>{1, 2, 1} : std::vector<int64_t>{2};
    for (const auto batch : batches) {
        SCOPED_TRACE(::testing::Message() << "KV batch=" << batch);
        auto set_input = [&](const std::string& id, int64_t input_batch, int64_t sequence, int64_t head_size) {
            const layout input_layout(dims(input_batch, sequence, head_size), dt, format::bfyx);
            auto memory = engine.allocate_memory(input_layout);
            std::vector<float> values(input_layout.count(), 0.f);
            if (id == "v") {
                for (size_t i = 0; i < values.size(); ++i) {
                    const auto b = order_kind == 2 ? (i / (4 * head_size)) % input_batch : i / (4 * sequence * head_size);
                    values[i] = 0.25f + 0.5f * static_cast<float>(b);
                }
            }
            if (dt == data_types::f32) {
                set_values(memory, values);
            } else if (dt == data_types::f16) {
                set_values(memory, std::vector<ov::float16>(values.begin(), values.end()));
            } else {
                set_values(memory, std::vector<ov::bfloat16>(values.begin(), values.end()));
            }
            net->set_input_data(id, memory);
        };
        // Keep Q at batch 1: no tiling or oversized backing allocations.
        set_input("q", 1, 2, 64);
        set_input("k", batch, 16, 64);
        set_input("v", batch, 16, 32);

        // Check the normal allocation before enqueue, so the regression fails without an out-of-bounds write.
        net->set_arguments();
        for (const auto& id : net->get_executed_primitive_ids()) {
            const auto instance = net->get_primitive(id);
            instance->reset_events();
            instance->prepare_primitive();
            if (id == "sdpa") {
                ASSERT_EQ(instance->get_impl()->m_manager->get_type_info(), ov::intel_gpu::ocl::SDPARef::get_type_info_static());
                ASSERT_EQ(instance->get_output_layout().get_shape(), (ov::Shape{static_cast<size_t>(batch), 4, 2, 32}));
                const auto descriptors = instance->get_impl()->get_internal_buffer_descs(*instance->get_impl_params());
                ASSERT_EQ(descriptors.size(), 1u);
                const data_types scratch_type = dt == data_types::bf16 ? data_types::f32 : dt;
                const size_t required_elements = static_cast<size_t>(batch) * 4 * 2 * 16;
                ASSERT_EQ(descriptors[0].m_layout.data_type, scratch_type);
                ASSERT_EQ(descriptors[0].m_layout.count(), required_elements) << "SDPARef scratch must cover the broadcast output batch";
                // Descriptor-only checks: output transposes and padding do not change the scratch geometry.
                // Do not enqueue the reference kernel with these synthetic output layouts.
                for (const auto& output_order : std::vector<std::vector<int64_t>>{{}, {0, 2, 1, 3}, {3, 0, 1, 2}}) {
                    auto params = *instance->get_impl_params();
                    auto desc = std::make_shared<scaled_dot_product_attention>(*params.typed_desc<scaled_dot_product_attention>());
                    desc->output_transpose_order = output_order;
                    params.desc = desc;
                    const auto shape = params.output_layouts[0].get_shape();
                    auto transposed_shape = shape;
                    for (size_t i = 0; i < output_order.size(); ++i) {
                        transposed_shape[i] = shape[output_order[i]];
                    }
                    params.output_layouts[0] = layout(transposed_shape, dt, format::bfyx, padding{{0, 0, 1, 1}, {0, 0, 1, 1}});
                    ASSERT_EQ(instance->get_impl()->get_internal_buffer_descs(params)[0].m_layout.count(), required_elements);
                }
                const auto& memories = instance->get_intermediates_memories();
                ASSERT_EQ(memories.size(), 1u);
                ASSERT_GE(memories[0]->size(), descriptors[0].m_layout.bytes_count());
            }
            instance->execute();
        }
        stream.finish();
        const auto kernels = sdpa->get_impl()->get_kernels_dump_info(*sdpa->get_impl_params()).get_entries();
        ASSERT_NE(kernels.find("sdpa_ref"), std::string::npos) << kernels;

        auto check_output = [&](const auto& output) {
            ASSERT_EQ(output.size(), static_cast<size_t>(batch) * 4 * 2 * 32);
            for (size_t i = 0; i < output.size(); ++i) {
                const float expected = 0.25f + 0.5f * static_cast<float>(i / (4 * 2 * 32));
                ASSERT_FLOAT_EQ(static_cast<float>(output[i]), expected) << "Output index " << i;
            }
        };
        const auto output = sdpa->output_memory_ptr();
        if (dt == data_types::f32) {
            check_output(mem_lock<float, mem_lock_type::read>(output, stream));
        } else if (dt == data_types::f16) {
            check_output(mem_lock<ov::float16, mem_lock_type::read>(output, stream));
        } else {
            check_output(mem_lock<ov::bfloat16, mem_lock_type::read>(output, stream));
        }
        for (const auto& id : net->get_executed_primitive_ids()) {
            net->get_primitive(id)->reset_flags();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(sdpa_ref_scratch,
                         sdpa_ref_scratch_test,
                         ::testing::Combine(::testing::Values(data_types::f32, data_types::f16, data_types::bf16),
                                            ::testing::Values(0, 1, 2),
                                            ::testing::Bool()));

// Compare FP16 optimized SDPA batch broadcasting with the FP32 reference implementation.

struct sdpa_broadcast_test_params {
    int q_batch;
    int k_batch;
    int v_batch;
    int num_heads;
    int seq_q;
    int seq_kv;
    int head_size;
    bool dynamic_batch;  // batch dim is dynamic in the network (tensors are static)
    int order_kind;      // 0: [B,H,L,E] default order; 1: [B,L,H,E] order {0,2,1,3}; 2: [L,B,H,E] order {1,2,0,3}

    sdpa_broadcast_test_params(int qb, int kb, int vb, int nh, int sq, int sk, int hs, bool dyn_batch, int kind)
        : q_batch(qb), k_batch(kb), v_batch(vb), num_heads(nh), seq_q(sq), seq_kv(sk), head_size(hs),
          dynamic_batch(dyn_batch), order_kind(kind) {}
};

struct sdpa_broadcast_test : public ::testing::TestWithParam<sdpa_broadcast_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    // Select SDPARef through a dynamic V head size; verify the actual implementation on both paths.
    cldnn::memory::ptr run_broadcast_network(const cldnn::layout& q_layout,
                                             const cldnn::layout& k_layout,
                                             const cldnn::layout& v_layout,
                                             cldnn::memory::ptr q_mem,
                                             cldnn::memory::ptr k_mem,
                                             cldnn::memory::ptr v_mem,
                                             const std::string& impl,
                                             int order_kind) {
        auto& engine = get_test_engine();
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        auto v_shape = v_layout.get_partial_shape();
        if (impl == "sdpa_ref")
            v_shape[3] = ov::Dimension::dynamic();
        topo.add(input_layout("v", cldnn::layout(v_shape, v_layout.data_type, v_layout.format)));

        const std::vector<std::vector<int64_t>> orders = {{0, 1, 2, 3}, {0, 2, 1, 3}, {1, 2, 0, 3}};
        const std::vector<int64_t>& in_order = orders.at(order_kind);
        auto sdpa_prim = scaled_dot_product_attention("sdpa",
                                                      {input_info("q"), input_info("k"), input_info("v")},
                                                      false,
                                                      -1,
                                                      in_order,
                                                      in_order,
                                                      in_order,
                                                      {0, 1, 2, 3},
                                                      {},
                                                      false);
        topo.add(sdpa_prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, q_layout.data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (!impl.empty() && impl != "sdpa_ref") {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                {"sdpa", {format::type::bfyx, impl}}}));
        }

        auto net = get_network(engine, topo, config, get_test_stream_ptr(), false);
        // A removable output reorder may rename the SDPA node to "result".
        std::string sdpa_id;
        for (const auto& info : net->get_primitives_info()) {
            if (info.type_id == "scaled_dot_product_attention")
                sdpa_id = info.original_id;
        }
        EXPECT_FALSE(sdpa_id.empty()) << "Expected an SDPA primitive";
        if (sdpa_id.empty())
            return nullptr;
        auto verify_manager = [&]() {
            const auto& manager = net->get_primitive(sdpa_id)->get_impl()->m_manager;
            const auto& expected_type =
                impl == "sdpa_ref" ? ov::intel_gpu::ocl::SDPARef::get_type_info_static() : ov::intel_gpu::ocl::SDPAOpt::get_type_info_static();
            EXPECT_EQ(manager->get_type_info(), expected_type) << "Unexpected SDPA implementation";
        };
        verify_manager();
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        auto result = net->execute().at("result").get_memory();
        verify_manager();
        const auto instance = net->get_primitive(sdpa_id);
        const auto executed_kernels = instance->get_impl()->get_kernels_dump_info(*instance->get_impl_params()).get_entries();
        const bool is_reference = impl == "sdpa_ref";
        EXPECT_EQ(executed_kernels.find("sdpa_ref") != std::string::npos, is_reference) << "Executed kernels: " << executed_kernels;
        EXPECT_EQ(executed_kernels.find("sdpa_micro") != std::string::npos || executed_kernels.find("sdpa_opt") != std::string::npos, !is_reference)
            << "Executed kernels: " << executed_kernels;
        RecordProperty(is_reference ? "reference_kernels" : "optimized_kernels", executed_kernels);
        EXPECT_EQ(result->get_layout().data_type, q_layout.data_type);
        return result;
    }

    void check_broadcast() {
        const auto& p = GetParam();
        auto& engine = get_test_engine();

        auto dims = [&](int64_t b, int64_t seq) {
            switch (p.order_kind) {
            case 1: return ov::PartialShape{b, seq, p.num_heads, p.head_size};
            case 2: return ov::PartialShape{seq, b, p.num_heads, p.head_size};
            default: return ov::PartialShape{b, p.num_heads, seq, p.head_size};
            }
        };
        const auto q_layout = cldnn::layout(dims(p.q_batch, p.seq_q), data_types::f16, format::bfyx);
        const auto k_layout = cldnn::layout(dims(p.k_batch, p.seq_kv), data_types::f16, format::bfyx);
        const auto v_layout = cldnn::layout(dims(p.v_batch, p.seq_kv), data_types::f16, format::bfyx);
        const auto net_q_layout = p.dynamic_batch ? cldnn::layout(dims(-1, p.seq_q), data_types::f16, format::bfyx) : q_layout;
        const auto net_k_layout = p.dynamic_batch ? cldnn::layout(dims(-1, p.seq_kv), data_types::f16, format::bfyx) : k_layout;
        const auto net_v_layout = p.dynamic_batch ? cldnn::layout(dims(-1, p.seq_kv), data_types::f16, format::bfyx) : v_layout;

        // Bound logits to reduce FP16 rounding sensitivity; V values distinguish logical batches.
        auto q_data = rg.generate_random_1d<ov::float16>(q_layout.count(), -1.0f, 1.0f);
        auto k_data = rg.generate_random_1d<ov::float16>(k_layout.count(), -1.0f, 1.0f);
        auto v_data = rg.generate_random_1d<ov::float16>(v_layout.count(), 0.25f, 0.75f);

        const auto v_shape = v_layout.get_shape();
        for (size_t i = 0; i < v_data.size(); ++i) {
            const size_t batch = p.order_kind == 2 ? (i / (v_shape[2] * v_shape[3])) % v_shape[1] : i / (v_shape[1] * v_shape[2] * v_shape[3]);
            v_data[i] = ov::float16(static_cast<float>(v_data[i]) + 0.5f * static_cast<float>(batch));
        }

        // Fill the space after the real Q/K/V data with NaN, so any read past the batch shows up as NaN.
        const int max_batch = std::max({p.q_batch, p.k_batch, p.v_batch});
        std::vector<cldnn::memory::ptr> keep_alive;
        auto make_guarded = [&](const cldnn::layout& real_layout, const std::vector<ov::float16>& data) {
            auto shape = real_layout.get_shape();
            shape[p.order_kind == 2 ? 1 : 0] = static_cast<size_t>(max_batch);
            cldnn::layout big_layout(shape, real_layout.data_type, real_layout.format);
            auto big = engine.allocate_memory(big_layout);
            std::vector<ov::float16> padded(big_layout.count(), ov::float16(std::numeric_limits<float>::quiet_NaN()));
            std::copy(data.begin(), data.end(), padded.begin());
            set_values(big, padded);
            keep_alive.push_back(big);
            return engine.reinterpret_buffer(*big, real_layout);
        };
        auto q_mem = make_guarded(q_layout, q_data);
        auto k_mem = make_guarded(k_layout, k_data);
        auto v_mem = make_guarded(v_layout, v_data);

        // sdpa_micro on systolic devices, sdpa_opt otherwise (micro is disabled on xe3p for head_size <= 64).
        auto& device_info = engine.get_device_info();
        const bool micro_eligible = device_info.supports_immad && !(device_info.arch == cldnn::gpu_arch::xe3p && p.head_size <= 64);
        const std::string opt_impl = micro_eligible ? "sdpa_micro" : "sdpa_opt";

        // Static batches that are neither equal nor 1 cannot be broadcast: kernel generation must reject them.
        auto batches_conflict = [](int a, int b) { return a != 1 && b != 1 && a != b; };
        if (!p.dynamic_batch && (batches_conflict(p.q_batch, p.k_batch) || batches_conflict(p.q_batch, p.v_batch) ||
                                 batches_conflict(p.k_batch, p.v_batch))) {
            EXPECT_ANY_THROW(run_broadcast_network(net_q_layout, net_k_layout, net_v_layout, q_mem, k_mem, v_mem, opt_impl, p.order_kind));
            return;
        }

        // Compute the reference in FP32 from the exact FP16 input values, not unrounded random values.
        auto as_f32_layout = [](cldnn::layout input) {
            input.data_type = data_types::f32;
            return input;
        };
        auto make_f32_memory = [&](const cldnn::layout& input_layout, const std::vector<ov::float16>& input_data) {
            auto memory = engine.allocate_memory(as_f32_layout(input_layout));
            set_values(memory, std::vector<float>(input_data.begin(), input_data.end()));
            return memory;
        };
        auto ref_q_mem = make_f32_memory(q_layout, q_data);
        auto ref_k_mem = make_f32_memory(k_layout, k_data);
        auto ref_v_mem = make_f32_memory(v_layout, v_data);
        auto mem_ref = run_broadcast_network(as_f32_layout(net_q_layout),
                                             as_f32_layout(net_k_layout),
                                             as_f32_layout(net_v_layout),
                                             ref_q_mem,
                                             ref_k_mem,
                                             ref_v_mem,
                                             "sdpa_ref",
                                             p.order_kind);
        auto mem_opt = run_broadcast_network(net_q_layout, net_k_layout, net_v_layout, q_mem, k_mem, v_mem, opt_impl, p.order_kind);

        ASSERT_NE(mem_ref, nullptr);
        ASSERT_NE(mem_opt, nullptr);
        cldnn::mem_lock<float, mem_lock_type::read> ref_ptr(mem_ref, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(mem_opt, get_test_stream());

        ASSERT_EQ(mem_ref->get_layout().get_shape(), mem_opt->get_layout().get_shape());
        ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
        // Keep the reference away from zero so invalid reads cannot hide within the tolerance.
        float max_abs_ref = 0.f;
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            ASSERT_TRUE(std::isfinite(static_cast<float>(ref_ptr[i]))) << "Reference output is not finite at index " << i;
            max_abs_ref = std::max(max_abs_ref, std::abs(static_cast<float>(ref_ptr[i])));
        }
        ASSERT_GT(max_abs_ref, 0.3f) << "Reference output too small: the test would not detect wrong K/V reads";
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            ASSERT_TRUE(std::isfinite(static_cast<float>(opt_ptr[i]))) << "Optimized output is not finite at index " << i;
            ASSERT_NEAR(static_cast<float>(opt_ptr[i]), static_cast<float>(ref_ptr[i]), 1e-2f)
                << "Broadcast SDPA mismatch at index " << i
                << ", opt=" << static_cast<float>(opt_ptr[i])
                << " ref=" << static_cast<float>(ref_ptr[i])
                << " (Q batch=" << p.q_batch << " K batch=" << p.k_batch << " V batch=" << p.v_batch << ")"
                << std::endl;
        }
    }
};

TEST_P(sdpa_broadcast_test, matches_reference) {
    check_broadcast();
}

INSTANTIATE_TEST_SUITE_P(
    sdpa_broadcast,
    sdpa_broadcast_test,
    ::testing::Values(
        // K/V batch = 1 (K and V broadcast, then each alone), default layout [B,H,L,E]
        sdpa_broadcast_test_params(2, 1, 1, 16, 1, 1024, 64, true, 0),
        sdpa_broadcast_test_params(2, 1, 1, 16, 64, 1024, 64, true, 0),
        sdpa_broadcast_test_params(2, 1, 1, 16, 1, 1024, 64, false, 0),
        sdpa_broadcast_test_params(2, 2, 1, 16, 1, 1024, 64, true, 0),
        sdpa_broadcast_test_params(2, 1, 2, 16, 1, 1024, 64, true, 0),
        // transposed layout [B,L,H,E] with order {0,2,1,3}
        sdpa_broadcast_test_params(2, 1, 1, 16, 64, 1024, 64, true, 1),
        sdpa_broadcast_test_params(2, 1, 1, 16, 1, 1024, 64, false, 1),
        // layout [L,B,H,E] with order {1,2,0,3}: the logical batch is not layout dim 0
        sdpa_broadcast_test_params(2, 1, 1, 16, 1, 1024, 64, true, 2),
        sdpa_broadcast_test_params(2, 1, 1, 16, 1, 1024, 64, false, 2),
        sdpa_broadcast_test_params(2, 2, 2, 16, 1, 1, 64, false, 2),
        // incompatible static batches are rejected
        sdpa_broadcast_test_params(4, 2, 2, 16, 64, 1024, 64, false, 0),
        // Q batch 1 is broadcast to K/V batch
        sdpa_broadcast_test_params(1, 2, 2, 16, 1, 1024, 64, true, 0),
        sdpa_broadcast_test_params(1, 2, 2, 16, 64, 1024, 64, true, 0),
        sdpa_broadcast_test_params(1, 2, 2, 16, 1, 1024, 64, false, 0),
        sdpa_broadcast_test_params(1, 4, 1, 16, 1, 1024, 64, true, 0),
        sdpa_broadcast_test_params(1, 1, 4, 16, 1, 1024, 64, true, 0),
        sdpa_broadcast_test_params(1, 2, 2, 16, 1, 1024, 64, true, 2),
        sdpa_broadcast_test_params(1, 2, 2, 16, 1, 1024, 64, false, 1)));

} // namespace

// The 2D block IO gates of sdpa_ocl, on padded layouts. Host-only: no device is involved.
//
// A paged-attention Q/K/V is rank 2 [tokens, heads * head], so feature padding moves the base and the pitch of
// every head, which the rank-4 argument ("the pitch is a multiple of the row") does not cover. The strict tier
// (block2d_layout_ok: no base repair) must prove a 64 B aligned base and pitch; the fixup tier
// (block2d_layout_fixup_ok: the kernel rounds the base down at run time) needs both to be multiples of 16 B,
// which it can only prove for static padding. A dynamic pad is taken on trust in the fixup tier (documented
// precondition) and refused by the strict one.
namespace {
using ov::intel_gpu::ocl::sdpa_ocl_utils::block2d_layout_fixup_ok;
using ov::intel_gpu::ocl::sdpa_ocl_utils::block2d_layout_ok;

// [?, 128] elements (f16: 2 heads of 64, row_bytes 128), FEATURE padded by before / after elements (0 / 0 = unpadded).
layout rank2_input(int before, int after, bool dynamic = false, data_types dt = data_types::f16, ov::Dimension feature = ov::Dimension(128)) {
    layout l{ov::PartialShape{ov::Dimension(-1), feature}, dt, format::bfyx};
    l.data_padding._lower_size[1] = before;
    l.data_padding._upper_size[1] = after;
    if (dynamic)
        l.data_padding._dynamic_dims_mask[1] = 1;
    return l;
}

// f16 [1, 8, 256, 64] (row_bytes 128) with one axis (index 2 = Y, 3 = X) padded.
layout rank4_input(size_t axis, int before, int after, bool dynamic = false) {
    layout l{ov::PartialShape{1, 8, 256, 64}, data_types::f16, format::bfyx};
    l.data_padding._lower_size[axis] = before;
    l.data_padding._upper_size[axis] = after;
    if (dynamic)
        l.data_padding._dynamic_dims_mask[axis] = 1;
    return l;
}

struct block2d_gate_row {
    const char* name;
    layout l;
    size_t row_bytes;
    bool strict;
    bool fixup;
};
}  // namespace

TEST(sdpa_block2d_gate, padded_layouts) {
    const std::vector<block2d_gate_row> rows = {
        // Rank 2 (paged attention), static padding: ld = 128 + before + after elements.
        {"unpadded", rank2_input(0, 0), 128, true, true},
        {"static 64 B / 64 B", rank2_input(32, 32), 128, true, true},
        {"static 0 / 64 B", rank2_input(0, 32), 128, true, true},
        {"static 64 B / 0", rank2_input(32, 0), 128, true, true},
        {"static 64 B / 32 B: pitch 352 B", rank2_input(32, 16), 128, false, true},
        {"static 16 B / 16 B: base and pitch off", rank2_input(8, 8), 128, false, true},
        // One condition at a time: only the start of the first head (32 B, stride 320 B), only the fixup start (4 B, stride 272 B), only the stride.
        {"static 32 B / 32 B: only the start breaks strict", rank2_input(16, 16), 128, false, true},
        {"static 4 B / 12 B: start 4 B, stride 272 B", rank2_input(2, 6), 128, false, false},
        {"static 0 / 4 B: only the stride is off", rank2_input(0, 2), 128, false, false},
        {"static 0 / 16 B: pitch 272 B", rank2_input(0, 8), 128, false, true},
        {"static 16 B / 0: pitch 272 B", rank2_input(8, 0), 128, false, true},
        {"static 4 B / 4 B: pitch 264 B", rank2_input(2, 2), 128, false, false},
        // Rank 2, dynamic padding: unprovable, so never the strict tier.
        {"dynamic", rank2_input(0, 0, true), 128, false, true},
        // A dynamic mask wins over static sizes; a feature dimension that is not static cannot be proven either.
        {"static 64 B / 64 B but dynamic", rank2_input(32, 32, true), 128, false, true},
        {"static pad, dynamic feature dim", rank2_input(0, 8, false, data_types::f16, ov::Dimension(-1)), 128, false, false},
        // The element size scales the byte arithmetic: f32 pads of 16 elements are 64 B, of 4 are 16 B, of 1 are 4 B; i8 of 16 are 16 B.
        {"f32 64 B / 64 B", rank2_input(16, 16, false, data_types::f32), 256, true, true},
        {"f32 16 B / 16 B", rank2_input(4, 4, false, data_types::f32), 256, false, true},
        {"f32 4 B / 4 B", rank2_input(1, 1, false, data_types::f32), 256, false, false},
        {"i8 16 B / 16 B", rank2_input(16, 16, false, data_types::i8), 128, false, true},
        // Row sizes: 64 B is the smallest surface, 80 B is fixup-only.
        {"unpadded, 64 B rows", rank2_input(0, 0), 64, true, true},
        {"unpadded, 80 B rows", rank2_input(0, 0), 80, false, true},
        // Ranks other than 2 and 4 have no innermost axis the test can look at: padded is refused, unpadded is fine.
        {"rank 3 unpadded", layout{ov::PartialShape{8, 256, 64}, data_types::f16, format::bfyx}, 128, true, true},
        // The width / pitch floor of the surface still applies.
        {"unpadded, 32 B rows", rank2_input(0, 0), 32, false, false},
        {"static 64 B / 64 B, 32 B rows", rank2_input(32, 32), 32, false, false},
        // Rank 4 (plain SDPA): offsets are whole rows unless the innermost axis is padded.
        {"rank 4 unpadded", rank4_input(2, 0, 0), 128, true, true},
        {"rank 4 Y static", rank4_input(2, 3, 5), 128, true, true},
        {"rank 4 Y dynamic", rank4_input(2, 0, 0, true), 128, true, true},
        {"rank 4 X static", rank4_input(3, 8, 0), 128, false, false},
        {"rank 4 X dynamic", rank4_input(3, 0, 0, true), 128, false, false},
    };
    for (const auto& r : rows) {
        SCOPED_TRACE(r.name);
        EXPECT_EQ(block2d_layout_ok(r.l, r.row_bytes), r.strict) << "strict tier";
        EXPECT_EQ(block2d_layout_fixup_ok(r.l, r.row_bytes), r.fixup) << "fixup tier";
    }

    // A padded rank-3 layout: the innermost axis is Y, which the X test would skip (vacuously true).
    layout rank3{ov::PartialShape{8, 256, 64}, data_types::f16, format::bfyx};
    rank3.data_padding._lower_size[2] = 8;
    SCOPED_TRACE("rank 3 padded");
    EXPECT_FALSE(block2d_layout_ok(rank3, 128)) << "strict tier";
    EXPECT_FALSE(block2d_layout_fixup_ok(rank3, 128)) << "fixup tier";
}
