// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "impls/ocl_v2/sdpa/sdpa_opt.hpp"
#include "openvino/util/file_util.hpp"
#include "program_wrapper.h"
#include <array>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>

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
    int sequence_length_q;
    int sequence_length_kv;
    int batch;
    bool dynamic;
    bool use_scalar_scale_val;
    float scale_val;
    bool use_scalar_attn_mask;
    float attn_mask_val;
    data_types dt;

    // Constructor for basic tests (backward compatibility)
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool dynamic_shape, data_types dt = data_types::f16)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q),
          sequence_length_kv(seq_kv), batch(b), dynamic(dynamic_shape),
          use_scalar_scale_val(false), scale_val(1.0f), use_scalar_attn_mask(false),
          attn_mask_val(0.0f), dt(dt) {}

    // Constructor for advanced caching tests
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool use_scale, float scale, bool use_mask, float mask, data_types dt = data_types::f16)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q), sequence_length_kv(seq_kv),
          batch(b), dynamic(true), use_scalar_scale_val(use_scale),
          scale_val(scale), use_scalar_attn_mask(use_mask), attn_mask_val(mask), dt(dt) {}
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

struct onednn_sdpa_test_params {
    int batch;
    int num_heads;
    int sequence_length_q;
    int sequence_length_kv;
    int head_size;
    std::optional<float> scale_val;
    bool use_runtime_scale = false;
    bool use_runtime_mask = false;
    bool dynamic = false;
};

struct onednn_sdpa_gpu_test : public ::testing::TestWithParam<onednn_sdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void fill_random(const memory::ptr& mem) {
        const auto shape = mem->get_layout().get_shape();
        const size_t elements_num = ov::shape_size(shape);
        auto data = rg.generate_random_1d<ov::float16>(elements_num, -0.5f, 0.5f);
        set_values(mem, data);
    }

    static network::ptr build_network(const layout& q_layout,
                                      const layout& k_layout,
                                      const layout& v_layout,
                                      const std::optional<layout>& mask_layout,
                                      const std::optional<layout>& scale_layout,
                                      bool use_onednn,
                                      const std::optional<float>& scale_val) {
        auto& engine = get_test_engine();

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        if (mask_layout.has_value())
            topo.add(input_layout("mask", mask_layout.value()));
        if (scale_layout.has_value())
            topo.add(input_layout("scale", scale_layout.value()));

        std::vector<input_info> inputs{input_info("q"), input_info("k"), input_info("v")};
        if (mask_layout.has_value())
            inputs.emplace_back("mask");
        if (scale_layout.has_value())
            inputs.emplace_back("scale");

        auto sdpa = scaled_dot_product_attention("sdpa",
                                                 inputs,
                                                 false,
                                                 -1,
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {},
                                                 false);
        if (scale_val.has_value()) {
            sdpa.scale_val = scale_val.value();
        }

        topo.add(sdpa);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::use_onednn(use_onednn));
        if (use_onednn) {
            config.set_property(ov::intel_gpu::optimize_data(true));
#ifdef ENABLE_DEBUG_CAPS
            config.set_property(ov::intel_gpu::use_onednn_sdpa(true));
#endif
        }
        if (!use_onednn) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                {"sdpa", {format::type::bfyx, "sdpa_ref", impl_types::ocl}}
            }));
        }

        return get_network(engine, topo, config, get_test_stream_ptr(), false);
    }

    static void assert_onednn_sdpa_selected(const network::ptr& net) {
        auto inst = net->get_primitive("sdpa");
        ASSERT_NE(inst, nullptr);
        auto impl = inst->get_impl();
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->m_manager, nullptr);
        EXPECT_TRUE(impl->is_onednn());
        EXPECT_EQ(impl->m_manager->get_impl_type(), impl_types::onednn);
        EXPECT_STREQ(impl->m_manager->get_type_info().name, "onednn::sdpa");
    }

    static void assert_onednn_sdpa_not_selected(const network::ptr& net) {
        auto inst = net->get_primitive("sdpa");
        ASSERT_NE(inst, nullptr);
        auto impl = inst->get_impl();
        ASSERT_NE(impl, nullptr);
        EXPECT_FALSE(impl->is_onednn());
    }

    void execute() {
        const auto p = GetParam();
        auto& engine = get_test_engine();
#ifndef ENABLE_DEBUG_CAPS
        GTEST_SKIP() << "oneDNN SDPA debug option requires ENABLE_DEBUG_CAPS";
#endif
        if (!engine.get_device_info().supports_immad || engine.get_device_info().arch == gpu_arch::unknown) {
            GTEST_SKIP() << "oneDNN SDPA requires IMMAD-capable GPU with known architecture";
        }

        const layout q_layout({p.batch, p.num_heads, p.sequence_length_q, p.head_size}, data_types::f16, format::bfyx);
        const layout k_layout({p.batch, p.num_heads, p.sequence_length_kv, p.head_size}, data_types::f16, format::bfyx);
        const layout v_layout({p.batch, p.num_heads, p.sequence_length_kv, p.head_size}, data_types::f16, format::bfyx);
        const layout mask_layout({p.batch, p.num_heads, 1, p.sequence_length_kv}, data_types::f16, format::bfyx);
        const layout scale_layout({1}, data_types::f16, format::bfyx);

        const layout q_topology_layout = p.dynamic ? layout({-1, p.num_heads, -1, p.head_size}, data_types::f16, format::bfyx) : q_layout;
        const layout k_topology_layout = p.dynamic ? layout({-1, p.num_heads, -1, p.head_size}, data_types::f16, format::bfyx) : k_layout;
        const layout v_topology_layout = p.dynamic ? layout({-1, p.num_heads, -1, p.head_size}, data_types::f16, format::bfyx) : v_layout;
        const auto mask_topology_layout = p.use_runtime_mask ? std::optional<layout>(p.dynamic ? layout({-1, p.num_heads, 1, -1}, data_types::f16, format::bfyx) : mask_layout)
                                                            : std::nullopt;
        const auto scale_topology_layout = p.use_runtime_scale ? std::optional<layout>(p.dynamic ? layout({1}, data_types::f16, format::bfyx) : scale_layout)
                                                              : std::nullopt;

        auto q_mem = engine.allocate_memory(q_layout);
        auto k_mem = engine.allocate_memory(k_layout);
        auto v_mem = engine.allocate_memory(v_layout);
        auto mask_mem = p.use_runtime_mask ? engine.allocate_memory(mask_layout) : memory::ptr{};
        auto scale_mem = p.use_runtime_scale ? engine.allocate_memory(scale_layout) : memory::ptr{};

        fill_random(q_mem);
        fill_random(k_mem);
        fill_random(v_mem);
        if (mask_mem)
            fill_random(mask_mem);
        if (scale_mem) {
            std::vector<ov::float16> scale_data(scale_layout.count(), ov::float16(0.125f));
            set_values(scale_mem, scale_data);
        }

        auto onednn_net = build_network(q_topology_layout, k_topology_layout, v_topology_layout, mask_topology_layout, scale_topology_layout, true, p.scale_val);
        assert_onednn_sdpa_selected(onednn_net);

        auto set_inputs = [&](const network::ptr& net) {
            net->set_input_data("q", q_mem);
            net->set_input_data("k", k_mem);
            net->set_input_data("v", v_mem);
            if (mask_mem)
                net->set_input_data("mask", mask_mem);
            if (scale_mem)
                net->set_input_data("scale", scale_mem);
        };

        set_inputs(onednn_net);

        auto ref_net = build_network(q_topology_layout, k_topology_layout, v_topology_layout, mask_topology_layout, scale_topology_layout, false, p.scale_val);
        set_inputs(ref_net);

        auto ref_output = ref_net->execute().at("sdpa").get_memory();
        auto onednn_output = onednn_net->execute().at("sdpa").get_memory();

        mem_lock<ov::float16, mem_lock_type::read> ref_data(ref_output, get_test_stream());
        mem_lock<ov::float16, mem_lock_type::read> onednn_data(onednn_output, get_test_stream());

        ASSERT_EQ(ref_data.size(), onednn_data.size());

        float max_abs_diff = 0.0f;
        float mean_abs_diff = 0.0f;
        for (size_t idx = 0; idx < ref_data.size(); ++idx) {
            ASSERT_FALSE(std::isnan(static_cast<float>(ref_data[idx]))) << "NaN in OCL reference output at " << idx;
            ASSERT_FALSE(std::isnan(static_cast<float>(onednn_data[idx]))) << "NaN in oneDNN output at " << idx;

            const auto diff = std::abs(static_cast<float>(ref_data[idx]) - static_cast<float>(onednn_data[idx]));
            max_abs_diff = std::max(max_abs_diff, diff);
            mean_abs_diff += diff;
        }
        mean_abs_diff /= static_cast<float>(ref_data.size());

        EXPECT_LT(max_abs_diff, 0.06f);
        EXPECT_LT(mean_abs_diff, 0.01f);
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<onednn_sdpa_test_params>& info) {
        std::string result = "onednn_sdpa_" + std::to_string(info.param.batch) + "_" +
                             std::to_string(info.param.num_heads) + "_" +
                             std::to_string(info.param.sequence_length_q) + "_" +
                             std::to_string(info.param.sequence_length_kv) + "_" +
                             std::to_string(info.param.head_size);
        if (info.param.scale_val.has_value()) {
            result += "_scale_" + std::to_string(static_cast<int>(info.param.scale_val.value() * 1000));
        } else {
            result += "_default_scale";
        }
        if (info.param.use_runtime_scale) {
            result += "_runtime_scale";
        }
        if (info.param.use_runtime_mask) {
            result += "_runtime_mask";
        }
        if (info.param.dynamic) {
            result += "_dynamic";
        }
        return result;
    }
};

TEST_P(onednn_sdpa_gpu_test, selects_onednn_and_validates_output) {
    execute();
}

TEST(onednn_sdpa_gpu_validation_test, rejects_unsupported_runtime_scale_and_batch_broadcast) {
    auto& engine = get_test_engine();
#ifndef ENABLE_DEBUG_CAPS
    GTEST_SKIP() << "oneDNN SDPA debug option requires ENABLE_DEBUG_CAPS";
#endif
    if (!engine.get_device_info().supports_immad || engine.get_device_info().arch == gpu_arch::unknown) {
        GTEST_SKIP() << "oneDNN SDPA requires IMMAD-capable GPU with known architecture";
    }

    const layout q_layout({2, 2, 4, 32}, data_types::f16, format::bfyx);
    const layout k_layout({2, 2, 6, 32}, data_types::f16, format::bfyx);
    const layout v_layout({2, 2, 6, 32}, data_types::f16, format::bfyx);
    const layout invalid_scale_layout({1, 2, 1, 1}, data_types::f16, format::bfyx);

    auto invalid_scale_net = onednn_sdpa_gpu_test::build_network(q_layout,
                                                                 k_layout,
                                                                 v_layout,
                                                                 std::nullopt,
                                                                 invalid_scale_layout,
                                                                 true,
                                                                 std::nullopt);
    onednn_sdpa_gpu_test::assert_onednn_sdpa_not_selected(invalid_scale_net);

    const layout broadcast_k_layout({1, 2, 6, 32}, data_types::f16, format::bfyx);
    const layout broadcast_v_layout({1, 2, 6, 32}, data_types::f16, format::bfyx);
    auto batch_broadcast_net = onednn_sdpa_gpu_test::build_network(q_layout,
                                                                  broadcast_k_layout,
                                                                  broadcast_v_layout,
                                                                  std::nullopt,
                                                                  std::nullopt,
                                                                  true,
                                                                  std::nullopt);
    onednn_sdpa_gpu_test::assert_onednn_sdpa_not_selected(batch_broadcast_net);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_onednn_sdpa_gpu_test,
    onednn_sdpa_gpu_test,
    ::testing::Values(
        onednn_sdpa_test_params{1, 2, 4, 6, 32, std::nullopt},
        onednn_sdpa_test_params{1, 2, 4, 6, 32, 0.125f},
        onednn_sdpa_test_params{1, 2, 4, 6, 32, std::nullopt, false, true, true},
        onednn_sdpa_test_params{1, 2, 4, 6, 32, std::nullopt, true, true, true}
    ),
    onednn_sdpa_gpu_test::PrintToStringParamName);
#endif

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
} // namespace
