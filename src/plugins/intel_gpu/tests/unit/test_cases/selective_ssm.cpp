// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/selective_ssm.hpp>
#include <vector>

#include "random_generator.hpp"
#include "selective_ssm_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct selective_ssm_test_params {
    int32_t batch;
    int32_t seq_len;
    int32_t num_heads;
    int32_t num_groups;
    int32_t head_dim;
    int32_t state_size;
    ov::element::Type precision;
};

struct selective_ssm_gpu_test : public ::testing::TestWithParam<selective_ssm_test_params> {
    tests::random_generator rg;

    template <typename T>
    static void run_reference(const std::vector<T>& A,
                              const std::vector<T>& dt,
                              const std::vector<T>& B,
                              const std::vector<T>& x,
                              const std::vector<T>& C,
                              std::vector<T>& state,
                              int32_t batch,
                              int32_t seq_len,
                              int32_t num_heads,
                              int32_t num_groups,
                              int32_t head_dim,
                              int32_t state_size,
                              std::vector<T>& output) {
        const int32_t heads_per_group = num_heads / num_groups;
        output.resize(static_cast<size_t>(batch) * seq_len * num_heads * head_dim);
        for (int32_t b = 0; b < batch; b++) {
            for (int32_t h = 0; h < num_heads; h++) {
                const int32_t g = h / heads_per_group;
                for (int32_t t = 0; t < seq_len; t++) {
                    const float dt_val = static_cast<float>(dt[(b * seq_len + t) * num_heads + h]);
                    const float dA = std::exp(static_cast<float>(A[h]) * dt_val);
                    for (int32_t p = 0; p < head_dim; p++) {
                        const float x_val = static_cast<float>(x[((b * seq_len + t) * num_heads + h) * head_dim + p]);
                        float acc = 0.f;
                        for (int32_t n = 0; n < state_size; n++) {
                            auto& s = state[((b * num_heads + h) * head_dim + p) * state_size + n];
                            const float new_state = static_cast<float>(s) * dA +
                                                    x_val * dt_val * static_cast<float>(B[((b * seq_len + t) * num_groups + g) * state_size + n]);
                            s = static_cast<T>(new_state);
                            acc += new_state * static_cast<float>(C[((b * seq_len + t) * num_groups + g) * state_size + n]);
                        }
                        output[((b * seq_len + t) * num_heads + h) * head_dim + p] = static_cast<T>(acc);
                    }
                }
            }
        }
    }

    template <typename T>
    void execute_t(const selective_ssm_test_params& p) {
        auto& engine = get_test_engine();
        const auto data_type = cldnn::element_type_to_data_type(p.precision);

        layout A_layout({p.num_heads}, data_type, format::bfyx);
        layout dt_layout({p.batch, p.seq_len, p.num_heads}, data_type, format::bfyx);
        layout BC_layout({p.batch, p.seq_len, p.num_groups, p.state_size}, data_type, format::bfyx);
        layout x_layout({p.batch, p.seq_len, p.num_heads, p.head_dim}, data_type, format::bfyx);
        layout state_layout({p.batch, p.num_heads, p.head_dim, p.state_size}, data_type, format::bfyx);

        auto A_mem = engine.allocate_memory(A_layout);
        auto dt_mem = engine.allocate_memory(dt_layout);
        auto B_mem = engine.allocate_memory(BC_layout);
        auto x_mem = engine.allocate_memory(x_layout);
        auto C_mem = engine.allocate_memory(BC_layout);
        auto state_mem = engine.allocate_memory(state_layout);

        auto A_data = rg.generate_random_1d<T>(A_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.2f), 256);
        auto dt_data = rg.generate_random_1d<T>(dt_mem->count(), static_cast<T>(0.f), static_cast<T>(0.5f), 256);
        auto B_data = rg.generate_random_1d<T>(B_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto x_data = rg.generate_random_1d<T>(x_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto C_data = rg.generate_random_1d<T>(C_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto state_data = rg.generate_random_1d<T>(state_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);

        set_values(A_mem, A_data);
        set_values(dt_mem, dt_data);
        set_values(B_mem, B_data);
        set_values(x_mem, x_data);
        set_values(C_mem, C_data);
        set_values(state_mem, state_data);

        topology topo;
        topo.add(input_layout("A", A_layout));
        topo.add(input_layout("dt", dt_layout));
        topo.add(input_layout("B", BC_layout));
        topo.add(input_layout("x", x_layout));
        topo.add(input_layout("C", BC_layout));
        topo.add(input_layout("state", state_layout));
        auto ssm_prim = selective_ssm("selective_ssm", {input_info("A"), input_info("dt"), input_info("B"), input_info("x"), input_info("C"), input_info("state")});
        ssm_prim.num_outputs = 2;
        topo.add(ssm_prim);
        topo.add(reorder("output", input_info("selective_ssm", 0), format::bfyx, data_type));
        topo.add(reorder("state_output", input_info("selective_ssm", 1), format::bfyx, data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);
        network->set_input_data("A", A_mem);
        network->set_input_data("dt", dt_mem);
        network->set_input_data("B", B_mem);
        network->set_input_data("x", x_mem);
        network->set_input_data("C", C_mem);
        network->set_input_data("state", state_mem);
        auto outputs = network->execute();

        std::vector<T> ref_output;
        auto ref_state = state_data;
        run_reference(A_data, dt_data, B_data, x_data, C_data, ref_state, p.batch, p.seq_len, p.num_heads, p.num_groups, p.head_dim, p.state_size, ref_output);

        auto out_mem = outputs.at("output").get_memory();
        auto state_out_mem = outputs.at("state_output").get_memory();
        cldnn::mem_lock<T, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
        cldnn::mem_lock<T, mem_lock_type::read> state_ptr(state_out_mem, get_test_stream());

        const float tol = p.precision == ov::element::f32 ? 1e-4f : 0.03f;
        for (size_t i = 0; i < ref_output.size(); i++) {
            ASSERT_NEAR(static_cast<float>(out_ptr[i]), static_cast<float>(ref_output[i]), tol) << "output idx=" << i;
        }
        for (size_t i = 0; i < ref_state.size(); i++) {
            ASSERT_NEAR(static_cast<float>(state_ptr[i]), static_cast<float>(ref_state[i]), tol) << "state idx=" << i;
        }
    }

    void execute(const selective_ssm_test_params& p) {
        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p);
        } else {
            execute_t<float>(p);
        }
    }
};

TEST_P(selective_ssm_gpu_test, basic) {
    execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke_selective_ssm_gpu_test,
                         selective_ssm_gpu_test,
                         ::testing::Values(selective_ssm_test_params{1, 4, 4, 2, 8, 8, ov::element::f32},
                                           selective_ssm_test_params{2, 3, 4, 1, 8, 16, ov::element::f32},
                                           selective_ssm_test_params{1, 4, 4, 2, 8, 8, ov::element::f16},
                                           selective_ssm_test_params{1, 8, 64, 1, 64, 128, ov::element::f32},
                                           selective_ssm_test_params{1, 8, 64, 1, 64, 128, ov::element::f16}));

}  // namespace
