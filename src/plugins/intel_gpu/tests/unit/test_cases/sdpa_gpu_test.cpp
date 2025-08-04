// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "openvino/util/file_util.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include "scaled_dot_product_attention_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;

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

    // Constructor for basic tests (backward compatibility)
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool dynamic_shape)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q),
          sequence_length_kv(seq_kv), batch(b), dynamic(dynamic_shape),
          use_scalar_scale_val(false), scale_val(1.0f), use_scalar_attn_mask(false),
          attn_mask_val(0.0f) {}

    // Constructor for advanced caching tests
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool use_scale, float scale, bool use_mask, float mask)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q), sequence_length_kv(seq_kv),
          batch(b), dynamic(true), use_scalar_scale_val(use_scale),
          scale_val(scale), use_scalar_attn_mask(use_mask), attn_mask_val(mask) {}
};

struct sdpa_gpu_test : public ::testing::TestWithParam<sdpa_test_params> {
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
            float attn_mask_val = 0.0f) {
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
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

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
            input0_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input1_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_layout = cldnn::layout({ -1, -1}, data_types::f16, format::bfyx);
            } else {
                input3_layout = cldnn::layout({-1, num_heads, -1, -1}, data_types::f16, format::bfyx);
            }

            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f32, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, data_types::f16, format::bfyx);
            }
        } else {
            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, data_types::f16, format::bfyx);
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

        load_input(input0, 0);
        load_input(input1, 1);
        load_input(input2, 2);
        load_input(input3, 3);

        auto [mem_ref_ptr, net_ref_ptr] = run_network(is_caching_test, false,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(is_caching_test, true,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val);

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

    float cosineSimilarity(cldnn::mem_lock<ov::float16, mem_lock_type::read>& vec1, cldnn::mem_lock<ov::float16, mem_lock_type::read>& memLockVec2) {
        if (vec1.size() != memLockVec2.size()) {
            return -1.0f;
        }

        float dotProduct = std::inner_product(vec1.begin(), vec1.end(), memLockVec2.begin(), 0.0f);

        float magnitude1 = std::sqrt(std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0f));
        float magnitude2 = std::sqrt(std::inner_product(memLockVec2.begin(), memLockVec2.end(), memLockVec2.begin(), 0.0f));

        if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
            return -1.0f;
        }

        return dotProduct / (magnitude1 * magnitude2);
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
        sdpa_test_params{64, 32, 128, 128, 2, false, 1.0f, true, 0.5f}     // attn_mask only
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
#endif
} // namespace
