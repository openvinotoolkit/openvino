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
    int32_t value_num_heads;
    int32_t head_size;
    ov::element::Type precision;

    gated_delta_net_test_params(int batch, int t, int num_heads, int value_num_heads, int head_size, ov::element::Type precision)
        : batch(batch), t(t), num_heads(num_heads), value_num_heads(value_num_heads), head_size(head_size), precision(precision) {}
};

struct gated_delta_net_gpu_test : public ::testing::TestWithParam<gated_delta_net_test_params> {
    tests::random_generator rg;
    size_t B = 0;
    size_t T = 0;
    size_t H = 0;
    size_t HK = 0;
    size_t K = 0;
    size_t V = 0;

    void SetUp() override {
        const auto& params = this->GetParam();
        B = params.batch;
        T = params.t;
        HK = params.num_heads;
        H = params.value_num_heads;
        K = params.head_size;
        V = params.head_size;
        const std::string seed = "gated_delta_net_" + std::to_string(B) + "_" + std::to_string(T) + "_" +
                                 std::to_string(H) + "_" + std::to_string(K) + "_" + params.precision.to_string();
        rg.set_seed(seed);
    }

    template<typename T>
    void load_input(cldnn::memory::ptr mem, size_t idx, std::vector<T>& input_data) {
        set_values(mem, input_data);
    }

    float dot_product(float* a, float* b, size_t n) {
        float result = 0.0f;
        for (size_t i = 0; i < n; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    void scale(float *a, float scale, size_t n) {
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] * scale;
        }
    }

    void add(float *a, float* b, size_t n) {
        for (size_t i = 0; i < n; i++) {
            a[i] += b[i];
        }
    }

    void l2norm(float* a, size_t n) {
        float eps = 0.000001;
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += a[j] * a[j];
        }
        sum += eps;
        sum = 1 / sqrt(sum);
        for (int j = 0; j < n; j++) {
            a[j] = a[j] * sum;
        }
    }

    template <typename T>
    void run_reference(const std::vector<T>& q,
                       const std::vector<T>& k,
                       const std::vector<T>& v,
                       const std::vector<T>& g,
                       const std::vector<T>& beta,
                       const float attn_scale,
                       std::vector<T>& state,
                       std::vector<T>& output) {
        for (size_t i_b = 0; i_b < this->B; i_b++) {
            for (size_t i_h = 0; i_h < this->H; i_h++) {
                for (size_t i_v = 0; i_v < this->V; i_v++) {
                    float init_state[128] = {0};
                    float b_k[128] = {0};
                    float b_q[128] = {0};
                    // B, T, HK, K for Q/K and B, T, H, K for V
                    size_t BATCH_STRIDE_Q = this->HK * this->K * this->T;
                    size_t BATCH_STRIDE_K = this->HK * this->K * this->T;
                    size_t BATCH_STRIDE_V = this->H * this->K * this->T;
                    size_t HEAD_STRIDE = this->K;
                    size_t group_size = this->H / this->HK;
                    size_t i_hk = i_h / group_size;
                    const T* q_ptr = q.data() + i_b * BATCH_STRIDE_Q + i_hk * HEAD_STRIDE;
                    const T* k_ptr = k.data() + i_b * BATCH_STRIDE_K + i_hk * HEAD_STRIDE;
                    const T* v_ptr = v.data() + i_b * BATCH_STRIDE_V + i_h * HEAD_STRIDE;
                    // B, H, K, V
                    for (size_t j = 0; j < this->K; j++) {
                        init_state[j] = state[i_b * this->H * this->V * this->K + i_h * this->V * this->K + j * this->V + i_v];
                    }
                    for (size_t i = 0; i < this->T; i++) {
                        // g: B, T, H
                        size_t G_B_STRIDE = this->T * this->H;
                        float b_g = g[i_b * G_B_STRIDE + i * this->H + i_h];
                        float b_beta = beta[i_b * G_B_STRIDE + i * this->H + i_h];
                        b_g = exp(b_g);
                        for (int j = 0; j < this->K; j++) {
                            b_k[j] = k_ptr[i * this->K * this->HK + j];
                            b_q[j] = q_ptr[i * this->K * this->HK + j];
                        }
                        
                        l2norm(b_k, this->K);
                        l2norm(b_q, this->K);

                        scale(b_q, attn_scale, this->K);

                        // h0 * g
                        scale(init_state, b_g, this->K);
                        float h_k = dot_product(init_state, b_k, this->K);
                        float b_v = v_ptr[i_v + i * this->V * this->H];
                        b_v -= h_k;
                        // b_v * b_k
                        b_v *= b_beta;
                        scale(b_k, b_v, this->K);
                        // h = h0 + update
                        add(init_state, b_k, this->K);
                        float b_output  = dot_product(init_state, b_q, this->K);
                        // B, T, H, V
                        output[i_b * this->T * this->H * this->V + i * this->H * this->V + i_h * this->V + i_v] = b_output;
                    }
                    // B, H, K, V
                    for (size_t j = 0; j < this->K; j++) {
                        state[i_b * this->H * this->V * this->K + i_h * this->V * this->K + j * this->V + i_v] = init_state[j];
                    }
                }
            }
        }
    }

    topology create_topology(layout q_layout,
                             layout k_layout,
                             layout v_layout,
                             layout g_layout,
                             layout beta_layout,
                             layout state_layout,
                             data_types output_dt) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("state", state_layout));
        topo.add(input_layout("g", g_layout));
        topo.add(input_layout("beta", beta_layout));

        auto linear_attn_prim =
            gated_delta_net("gated_delta_net", {input_info("q"), input_info("k"), input_info("v"), input_info("state"), input_info("g"), input_info("beta")});
        topo.add(linear_attn_prim);
        topo.add(reorder("output", input_info("gated_delta_net", 0), format::bfyx, output_dt));
        // topo.add(reorder("states", input_info("gated_delta_net", 1), format::bfyx, data_types::f16));
        return topo;
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(topology &topo,
            cldnn::memory::ptr q_mem,
            cldnn::memory::ptr k_mem,
            cldnn::memory::ptr v_mem,
            cldnn::memory::ptr state_mem,
            cldnn::memory::ptr g_mem,
            cldnn::memory::ptr beta_mem,
            const bool is_caching_test) {
        auto& engine = get_test_engine();

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("state", state_mem);
        net->set_input_data("g", g_mem);
        net->set_input_data("beta", beta_mem);

        auto outputs = net->execute();
        auto output = outputs.at("output").get_memory();
        return {output, net};
    }

    template <typename T>
    void execute_t(gated_delta_net_test_params& p,
                   data_types data_type,
                   float tolerance,
                   const bool is_caching_test = false) {
        const auto batch = p.batch;
        const auto t = p.t;
        const auto qk_num_heads = p.num_heads;
        const auto v_num_heads = p.value_num_heads;
        const auto head_size = p.head_size;

        auto& engine = get_test_engine();

        // create topologies
        cldnn::layout q_dyn_layout({-1, -1, qk_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout k_dyn_layout({-1, -1, qk_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout v_dyn_layout({-1, -1, v_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout g_dyn_layout({-1, -1, v_num_heads}, data_type, format::bfyx);
        cldnn::layout beta_dyn_layout({-1, -1, v_num_heads}, data_type, format::bfyx);
        cldnn::layout state_dyn_layout({-1, v_num_heads, head_size, head_size}, data_type, format::bfyx);

        topology opt_topo = create_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, g_dyn_layout, g_dyn_layout, state_dyn_layout, data_type);

        // allocate memories
        cldnn::layout q_static_layout({batch, t, qk_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout k_static_layout({batch, t, qk_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout v_static_layout({batch, t, v_num_heads, head_size}, data_type, format::bfyx);
        cldnn::layout state_static_layout({batch, v_num_heads, head_size, head_size}, data_type, format::bfyx);
        cldnn::layout g_static_layout({batch, t, v_num_heads}, data_type, format::bfyx);
        cldnn::layout beta_static_layout({batch, t, v_num_heads}, data_type, format::bfyx);

        auto q_mem = engine.allocate_memory(q_static_layout);
        auto k_mem = engine.allocate_memory(k_static_layout);
        auto v_mem = engine.allocate_memory(v_static_layout);
        auto state_mem = engine.allocate_memory(state_static_layout);
        auto g_mem = engine.allocate_memory(g_static_layout);
        auto beta_mem = engine.allocate_memory(beta_static_layout);

        auto input_q = rg.generate_random_1d<T>(ov::shape_size(q_mem->get_layout().get_shape()), -1, 1);
        auto input_k = rg.generate_random_1d<T>(ov::shape_size(k_mem->get_layout().get_shape()), -1, 1);
        auto input_v = rg.generate_random_1d<T>(ov::shape_size(v_mem->get_layout().get_shape()), -1, 1);
        auto input_g = rg.generate_random_1d<T>(ov::shape_size(g_mem->get_layout().get_shape()), -1, 1);
        auto input_beta = rg.generate_random_1d<T>(ov::shape_size(beta_mem->get_layout().get_shape()), 0, 1);
        auto input_state = rg.generate_random_1d<T>(ov::shape_size(state_mem->get_layout().get_shape()), -1, 1);

        load_input(q_mem, 0, input_q);
        load_input(k_mem, 1, input_k);
        load_input(v_mem, 2, input_v);
        load_input(state_mem, 3, input_state);
        load_input(g_mem, 4, input_g);
        load_input(beta_mem, 5, input_beta);
        // execute networks
        auto [mem_opt_ptr, net_opt_ptr] = run_network(opt_topo,
                                        q_mem, k_mem, v_mem, state_mem, g_mem, beta_mem,
                                        is_caching_test);
        std::vector<T> ref_output(batch*t*v_num_heads*head_size);
        run_reference<T>(input_q, input_k, input_v, input_g, input_beta, 1/sqrt(this->K), input_state, ref_output);
        // validate results
        if (mem_opt_ptr) {
            cldnn::mem_lock<T, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
            ASSERT_EQ(mem_opt_ptr->count(), ref_output.size());
            for (size_t i = 0; i < mem_opt_ptr->count(); i++) {
                ASSERT_NEAR(opt_data[i], ref_output[i], tolerance) << " at index=" << i;
            }
        }

        if (state_mem) {
            ASSERT_EQ(state_mem->count(), input_state.size());
            cldnn::mem_lock<T, mem_lock_type::read> state_data(state_mem, get_test_stream());
            const float state_tolerance = tolerance;
            for (size_t i = 0; i < state_mem->count(); i++) {
                ASSERT_NEAR(state_data[i], input_state[i], state_tolerance) << " at index=" << i;
            }
        }
    }

    void execute(gated_delta_net_test_params& p, const bool is_caching_test = false) {
        auto cldnn_precision = cldnn::element_type_to_data_type(p.precision);
        float tolerance = 0.01f;
        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p, cldnn_precision, tolerance, is_caching_test);
            return;
        }

        if (p.precision == ov::element::f32) {
            execute_t<float>(p, cldnn_precision, 1e-3f, is_caching_test);
            return;
        }

        FAIL() << "Unsupported precision for linear attention test";
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<gated_delta_net_test_params>& info) {
         std::string result = "gated_delta_net_gpu_test_" + info.param.precision.to_string() + "_" + std::to_string(info.param.batch) + "_" +
             std::to_string(info.param.t) + "_" + std::to_string(info.param.num_heads) + "_" +
             std::to_string(info.param.value_num_heads) + "_" +
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

TEST_P(gated_delta_net_gpu_test, basic) {
    if (!check_cm_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p);
}

// TEST_P(gated_delta_net_gpu_test, basic_caching) {
//     if (!check_cm_available())
//         GTEST_SKIP();

//     auto p = GetParam();
//     execute(p, true);
// }

INSTANTIATE_TEST_SUITE_P(smoke_gated_delta_net_gpu_test,
    gated_delta_net_gpu_test,
    ::testing::Values(
        // B, T, H_QK, H, K, precision
        gated_delta_net_test_params{1, 1, 2, 2, 16, ov::element::f16},
        gated_delta_net_test_params{1, 8, 2, 2, 16, ov::element::f16},
        gated_delta_net_test_params{2, 8, 2, 2, 16, ov::element::f16},
        gated_delta_net_test_params{1, 8, 2, 2, 128, ov::element::f16},
        gated_delta_net_test_params{2, 8, 2, 2, 128, ov::element::f16},
        gated_delta_net_test_params{1, 8, 2, 4, 16, ov::element::f16},
        gated_delta_net_test_params{2, 8, 2, 4, 16, ov::element::f16},
        gated_delta_net_test_params{1, 8, 2, 8, 16, ov::element::f16},
        gated_delta_net_test_params{1, 8, 4, 8, 16, ov::element::f16},
        gated_delta_net_test_params{1, 8, 2, 4, 128, ov::element::f16},
        gated_delta_net_test_params{1, 1, 2, 2, 16, ov::element::f32},
        gated_delta_net_test_params{1, 8, 2, 2, 16, ov::element::f32},
        gated_delta_net_test_params{2, 8, 2, 2, 16, ov::element::f32},
        gated_delta_net_test_params{1, 8, 2, 2, 128, ov::element::f32},
        gated_delta_net_test_params{2, 8, 2, 2, 128, ov::element::f32},
        gated_delta_net_test_params{1, 8, 2, 4, 16, ov::element::f32},
        gated_delta_net_test_params{2, 8, 2, 4, 16, ov::element::f32},
        gated_delta_net_test_params{1, 8, 2, 8, 16, ov::element::f32},
        gated_delta_net_test_params{1, 8, 4, 8, 16, ov::element::f32},
        gated_delta_net_test_params{1, 8, 2, 4, 128, ov::element::f32}
    ),
    gated_delta_net_gpu_test::PrintToStringParamName
);

} // namespace