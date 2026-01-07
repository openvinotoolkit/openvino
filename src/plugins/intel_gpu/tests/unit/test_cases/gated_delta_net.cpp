// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/util/file_util.hpp"

#include <intel_gpu/runtime/debug_configuration.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <intel_gpu/primitives/gated_delta_net.hpp>
#include "gated_delta_net_inst.h"

#include <cstddef>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace cldnn;
using namespace ::tests;
namespace  {
struct gated_delta_net_test_params {
    int32_t batch;
    int32_t t;
    int32_t num_heads;
    int32_t head_size;

    gated_delta_net_test_params(int batch, int t, int num_heads, int head_size)
        : batch(batch), t(t), num_heads(num_heads), head_size(head_size) {}
};

struct gated_delta_net_gpu_test : public ::testing::TestWithParam<gated_delta_net_test_params> {
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

    // topology create_ref_topology(layout q_layout, layout k_layout, layout v_layout, layout attn_mask_layout) {
    //     topology topo;
    //     topo.add(input_layout("q", q_layout));
    //     topo.add(input_layout("k", k_layout));
    //     topo.add(input_layout("v", v_layout));
    //     topo.add(input_layout("attn", attn_mask_layout));  // "attention_mask"

    //     auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
    //         false, -1, order_q, order_k, order_v, {0, 1, 2}, {}, false);
    //     topo.add(sdpa_prim);

    //     topo.add(permute("permute_o", input_info("sdpa"), order_o2));

    //     topo.add(reorder("result",input_info("permute_o"), format::bfyx, data_types::f16));
    //     return topo;
    // };

    topology create_topology(layout q_layout, layout k_layout, layout v_layout, layout g_layout, layout beta_layout, layout state_layout) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("g", g_layout));
        topo.add(input_layout("beta", beta_layout));
        topo.add(input_layout("state", state_layout));

        auto linear_attn_prim =
            linear_attention("vlsdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("g"), input_info("beta"), input_info("state")});

        topo.add(linear_attn_prim);
        return topo;
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(topology &topo,
            cldnn::memory::ptr q_mem,
            cldnn::memory::ptr k_mem,
            cldnn::memory::ptr v_mem,
            cldnn::memory::ptr g_mem,
            cldnn::memory::ptr beta_mem,
            cldnn::memory::ptr state_mem,
            const bool is_caching_test) {
        auto& engine = get_test_engine();

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("g", g_mem);
        net->set_input_data("beta", beta_mem);
        net->set_input_data("state", state_mem);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(linear_attention_test_params& p, const bool is_caching_test = false) {
        const auto batch = p.head_size;
        const auto t = p.t;
        const auto num_heads = p.num_heads;
        const auto head_size = p.head_size;

        auto& engine = get_test_engine();

        // create topologies
        cldnn::layout q_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout g_dyn_layout({-1, -1, num_heads}, data_types::f16, format::bfyx);
        cldnn::layout beta_dyn_layout({-1, -1, num_heads}, data_types::f16, format::bfyx);
        cldnn::layout state_dyn_layout({-1, -1, head_size, head_size}, data_types::f16, format::bfyx);

        topology opt_topo = create_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, g_dyn_layout, g_dyn_layout, state_dyn_layout);

        // allocate memories
        cldnn::layout q_static_layout({batch, t, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_static_layout({batch, t, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_static_layout({batch, t, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout g_static_layout({batch, t, num_heads}, data_types::f16, format::bfyx);
        cldnn::layout beta_static_layout({batch, t, num_heads}, data_types::f16, format::bfyx);
        cldnn::layout state_static_layout({batch, t, head_size, head_size}, data_types::f16, format::bfyx);

        auto q_mem = engine.allocate_memory(q_static_layout);
        auto k_mem = engine.allocate_memory(k_static_layout);
        auto v_mem = engine.allocate_memory(v_static_layout);
        auto g_mem = engine.allocate_memory(g_static_layout);
        auto beta_mem = engine.allocate_memory(beta_static_layout);
        auto state_mem = engine.allocate_memory(state_static_layout);

        load_input(q_mem, 0);
        load_input(k_mem, 1);
        load_input(v_mem, 2);
        load_input(g_mem, 3);
        load_input(beta_mem, 4);
        load_input(state_mem, 5);

        // execute networks
        auto [mem_opt_ptr, net_opt_ptr] = run_network(opt_topo,
                                        q_mem, k_mem, v_mem, g_mem, beta_mem, state_mem,
                                        is_caching_test);

        // validate results
        // cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        // {
        //     for (size_t idx = 0; idx < ref_data.size(); idx++) {
        //         ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
        //     }
        //     auto ret = cosineSimilarity(ref_data, opt_data);
        //     ASSERT_GE(ret, 0.95f);
        // }
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<linear_attention_test_params>& info) {
        std::string result = "linear_attention_gpu_test_" + std::to_string(info.param.batch) + "_" +
               std::to_string(info.param.t) + "_" + std::to_string(info.param.num_heads) + "_" +
               std::to_string(info.param.head_size);

        return result;
    }

    static bool check_cm_available() {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        if (!cldnn::check_cm_jit_support(engine, config)) {
            return false;
        }

        return true;
    }
};

TEST_P(linear_attention_gpu_test, basic) {
    if (!check_cm_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p);
}

// TEST_P(linear_attention_gpu_test, basic_caching) {
//     if (!check_cm_available())
//         GTEST_SKIP();

//     auto p = GetParam();
//     execute(p, true);
// }

INSTANTIATE_TEST_SUITE_P(smoke_gated_delta_net_gpu_test,
    gated_delta_net_gpu_test,
    ::testing::Values(
        gated_delta_net_test_params{1, 1, 16, 16}
    ),
    gated_delta_net_gpu_test::PrintToStringParamName
);

} // namespace
