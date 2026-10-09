// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/selective_ssm.hpp>
#include <string>
#include <vector>

#include "selective_ssm_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr int32_t large_state_size = 32 * 1024 + 1;

enum class expected_ssm_impl { any, jit, fallback };

template <typename T>
std::vector<T> make_test_values(size_t count, float scale, float shift = 0.f) {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = static_cast<T>((static_cast<int32_t>(i % 11) - 5) * scale + shift);
    return values;
}

struct selective_ssm_test_params {
    int32_t batch;
    int32_t seq_len;
    int32_t num_heads;
    int32_t num_groups;
    int32_t head_dim;
    int32_t state_size;
    ov::element::Type precision;
    bool dynamic_shapes;
    bool caching_test = false;
    bool padded_layouts = false;
    int32_t iterations = 1;
    bool accumulation_test = false;
    float relative_output_tolerance = 5e-5f;
    expected_ssm_impl expected_impl = expected_ssm_impl::any;
};

selective_ssm_test_params expect_impl(selective_ssm_test_params params, expected_ssm_impl expected) {
    params.expected_impl = expected;
    return params;
}

struct selective_ssm_gpu_test : public ::testing::TestWithParam<selective_ssm_test_params> {
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
        std::vector<float> state_fp32(state.size());
        for (size_t i = 0; i < state.size(); ++i)
            state_fp32[i] = static_cast<float>(state[i]);

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
                            auto& s = state_fp32[((b * num_heads + h) * head_dim + p) * state_size + n];
                            const float new_state = s * dA + x_val * dt_val * static_cast<float>(B[((b * seq_len + t) * num_groups + g) * state_size + n]);
                            s = new_state;
                            acc += s * static_cast<float>(C[((b * seq_len + t) * num_groups + g) * state_size + n]);
                        }
                        output[((b * seq_len + t) * num_heads + h) * head_dim + p] = static_cast<T>(acc);
                    }
                }
            }
        }
        for (size_t i = 0; i < state.size(); ++i)
            state[i] = static_cast<T>(state_fp32[i]);
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

        const auto allocate = [&](const layout& requested_layout) {
            if (requested_layout.count() != 0)
                return engine.allocate_memory(requested_layout);
            auto dummy = engine.allocate_memory(layout{{1}, data_types::u8, format::bfyx});
            return engine.reinterpret_buffer(*dummy, requested_layout);
        };

        auto A_mem = allocate(A_layout);
        auto dt_mem = allocate(dt_layout);
        auto B_mem = allocate(BC_layout);
        auto x_mem = allocate(x_layout);
        auto C_mem = allocate(BC_layout);
        auto state_mem = allocate(state_layout);

        auto A_data = make_test_values<T>(A_mem->count(), -0.03f, -0.25f);
        auto dt_data = make_test_values<T>(dt_mem->count(), 0.007f, 0.08f);
        auto B_data = make_test_values<T>(B_mem->count(), 0.01f);
        auto x_data = make_test_values<T>(x_mem->count(), 0.015f);
        auto C_data = make_test_values<T>(C_mem->count(), 0.012f);
        auto state_data = make_test_values<T>(state_mem->count(), 0.008f);

        if (p.accumulation_test) {
            A_data.assign(A_data.size(), static_cast<T>(0.f));
            dt_data.assign(dt_data.size(), static_cast<T>(1.f));
            B_data.assign(B_data.size(), static_cast<T>(1.f));
            const float increment = p.precision == ov::element::bf16 ? 0.0546875f : 0.195068359375f;
            x_data.assign(x_data.size(), static_cast<T>(increment));
            C_data.assign(C_data.size(), static_cast<T>(1.f));
            state_data.assign(state_data.size(), static_cast<T>(0.f));
        }

        const auto set_non_empty = [](const memory::ptr& mem, const auto& values) {
            if (!values.empty())
                set_values(mem, values);
        };
        set_non_empty(A_mem, A_data);
        set_non_empty(dt_mem, dt_data);
        set_non_empty(B_mem, B_data);
        set_non_empty(x_mem, x_data);
        set_non_empty(C_mem, C_data);
        set_non_empty(state_mem, state_data);

        const auto dt_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, -1, p.num_heads}, data_type, format::bfyx} : dt_layout;
        const auto BC_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, -1, p.num_groups, p.state_size}, data_type, format::bfyx} : BC_layout;
        const auto x_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, -1, p.num_heads, p.head_dim}, data_type, format::bfyx} : x_layout;
        const auto state_input_layout =
            p.dynamic_shapes ? layout{ov::PartialShape{-1, p.num_heads, p.head_dim, p.state_size}, data_type, format::bfyx} : state_layout;

        topology topo;
        topo.add(input_layout("A", A_layout));
        topo.add(input_layout("dt", dt_input_layout));
        topo.add(input_layout("B", BC_input_layout));
        topo.add(input_layout("x", x_input_layout));
        topo.add(input_layout("C", BC_input_layout));
        topo.add(input_layout("state", state_input_layout));
        std::vector<input_info> ssm_inputs{input_info("A"), input_info("dt"), input_info("B"), input_info("x"), input_info("C"), input_info("state")};
        if (p.padded_layouts) {
            OPENVINO_ASSERT(!p.dynamic_shapes, "Padded-layout test requires static input layouts");
            const auto A_padding = padding(std::vector<ov::Dimension::value_type>{1}, std::vector<ov::Dimension::value_type>{2});
            topo.add(reorder("A_padded", input_info("A"), A_layout.with_padding(A_padding)));
            topo.add(reorder("dt_padded", input_info("dt"), dt_layout.with_padding(padding({1, 2, 1}, {2, 1, 3}))));
            topo.add(reorder("B_padded", input_info("B"), BC_layout.with_padding(padding({1, 2, 1, 3}, {2, 1, 2, 1}))));
            topo.add(reorder("x_padded", input_info("x"), x_layout.with_padding(padding({1, 2, 1, 3}, {2, 1, 2, 1}))));
            topo.add(reorder("C_padded", input_info("C"), BC_layout.with_padding(padding({1, 2, 1, 3}, {2, 1, 2, 1}))));
            topo.add(reorder("state_padded", input_info("state"), state_layout.with_padding(padding({1, 2, 1, 3}, {2, 1, 2, 1}))));
            ssm_inputs = {input_info("A_padded"),
                          input_info("dt_padded"),
                          input_info("B_padded"),
                          input_info("x_padded"),
                          input_info("C_padded"),
                          input_info("state_padded")};
        }
        auto ssm_prim = selective_ssm("selective_ssm", ssm_inputs);
        if (p.padded_layouts) {
            ssm_prim.output_paddings = {padding({1, 2, 1, 3}, {2, 1, 2, 1}), padding({1, 2, 1, 3}, {2, 1, 2, 1})};
        }
        topo.add(ssm_prim);
        topo.add(reorder("output", input_info("selective_ssm", 0), format::bfyx, data_type));
        topo.add(reorder("state_output", input_info("selective_ssm", 1), format::bfyx, data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), p.caching_test);
        if (p.expected_impl != expected_ssm_impl::any) {
            const auto primitive = network->get_primitive("selective_ssm");
            ASSERT_NE(primitive, nullptr);
            auto* const impl = primitive->get_impl();
            ASSERT_NE(impl, nullptr);
            const bool is_jit = impl->get_kernel_name().find("jit_") != std::string::npos;
            EXPECT_EQ(is_jit, p.expected_impl == expected_ssm_impl::jit) << "selected implementation: " << impl->get_kernel_name();
        }
        network->set_input_data("A", A_mem);
        network->set_input_data("dt", dt_mem);
        network->set_input_data("B", B_mem);
        network->set_input_data("x", x_mem);
        network->set_input_data("C", C_mem);
        network->set_input_data("state", state_mem);

        const float tol = p.precision == ov::element::f32 ? 2e-4f : (p.precision == ov::element::bf16 ? 0.08f : 0.03f);
        for (int32_t iteration = 0; iteration < p.iterations; iteration++) {
            auto outputs = network->execute();
            std::vector<T> ref_output;
            auto ref_state = state_data;
            run_reference(A_data,
                          dt_data,
                          B_data,
                          x_data,
                          C_data,
                          ref_state,
                          p.batch,
                          p.seq_len,
                          p.num_heads,
                          p.num_groups,
                          p.head_dim,
                          p.state_size,
                          ref_output);

            if (!ref_output.empty()) {
                auto out_mem = outputs.at("output").get_memory();
                ASSERT_NE(out_mem, nullptr);
                cldnn::mem_lock<T, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
                for (size_t i = 0; i < ref_output.size(); i++) {
                    const float reference = static_cast<float>(ref_output[i]);
                    const float output_tol = p.precision == ov::element::f32 ? std::max(tol, std::abs(reference) * p.relative_output_tolerance) : tol;
                    ASSERT_NEAR(static_cast<float>(out_ptr[i]), reference, output_tol) << "iteration=" << iteration << ", output idx=" << i;
                }
            }
            if (!ref_state.empty()) {
                auto state_out_mem = outputs.at("state_output").get_memory();
                ASSERT_NE(state_out_mem, nullptr);
                cldnn::mem_lock<T, mem_lock_type::read> state_ptr(state_out_mem, get_test_stream());
                for (size_t i = 0; i < ref_state.size(); i++) {
                    ASSERT_NEAR(static_cast<float>(state_ptr[i]), static_cast<float>(ref_state[i]), tol) << "iteration=" << iteration << ", state idx=" << i;
                }
            }
        }
    }

    void execute(const selective_ssm_test_params& p) {
        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p);
        } else if (p.precision == ov::element::bf16) {
            execute_t<ov::bfloat16>(p);
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
                         ::testing::Values(selective_ssm_test_params{1, 4, 4, 2, 8, 8, ov::element::f32, false},
                                           selective_ssm_test_params{2, 3, 4, 1, 8, 16, ov::element::f32, false},
                                           selective_ssm_test_params{1, 4, 4, 2, 8, 8, ov::element::f16, false},
                                           selective_ssm_test_params{1, 3, 2, 1, 4, 16, ov::element::bf16, false},
                                           expect_impl(selective_ssm_test_params{1, 1, 64, 1, 64, 128, ov::element::f32, false}, expected_ssm_impl::jit),
                                           selective_ssm_test_params{1, 1, 64, 1, 64, 128, ov::element::f16, false},
                                           selective_ssm_test_params{1, 1, 64, 1, 64, 128, ov::element::bf16, false},
                                           selective_ssm_test_params{1, 2, 64, 1, 64, 128, ov::element::f16, false},
                                           selective_ssm_test_params{1, 2, 64, 1, 64, 128, ov::element::bf16, false},
                                           selective_ssm_test_params{1, 8, 64, 1, 64, 128, ov::element::f32, false},
                                           selective_ssm_test_params{1, 8, 64, 1, 64, 128, ov::element::f16, false},
                                           selective_ssm_test_params{1, 16, 64, 1, 64, 128, ov::element::f16, false},
                                           selective_ssm_test_params{1, 32, 64, 1, 64, 128, ov::element::f32, false},
                                           selective_ssm_test_params{1, 64, 64, 1, 64, 128, ov::element::f32, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 256, ov::element::f32, false},
                                           selective_ssm_test_params{1, 16, 2, 1, 4, 256, ov::element::f32, false},
                                           selective_ssm_test_params{1, 16, 2, 1, 4, 256, ov::element::f16, false},
                                           selective_ssm_test_params{1, 8, 2, 1, 4, 256, ov::element::bf16, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 512, ov::element::f32, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 512, ov::element::f16, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 512, ov::element::bf16, false},
                                           expect_impl(selective_ssm_test_params{1, 2, 2, 1, 2, 513, ov::element::f32, false}, expected_ssm_impl::fallback),
                                           expect_impl(selective_ssm_test_params{2, 3, 4, 2, 4, 16, ov::element::f32, true}, expected_ssm_impl::fallback),
                                           selective_ssm_test_params{1, 7, 4, 2, 5, 31, ov::element::f16, true},
                                           selective_ssm_test_params{1, 5, 4, 2, 8, 17, ov::element::f32, false, true},
                                           expect_impl(selective_ssm_test_params{1, 5, 4, 2, 3, 19, ov::element::f32, false, false, true},
                                                       expected_ssm_impl::fallback),
                                           selective_ssm_test_params{2, 0, 4, 2, 3, 17, ov::element::f16, true},
                                           selective_ssm_test_params{1, 0, 2, 1, 4, 9, ov::element::bf16, false},
                                           selective_ssm_test_params{2, 4, 4, 2, 3, 15, ov::element::f32, true, false, false, 3},
                                           selective_ssm_test_params{2, 32, 8, 4, 16, 64, ov::element::f16, false},
                                           selective_ssm_test_params{1, 128, 4, 2, 8, 32, ov::element::f16, false},
                                           selective_ssm_test_params{1, 128, 4, 2, 8, 32, ov::element::bf16, false},
                                           selective_ssm_test_params{1, 128, 1, 1, 1, 1, ov::element::f16, false, false, false, 1, true},
                                           selective_ssm_test_params{1, 128, 1, 1, 1, 1, ov::element::bf16, false, false, false, 1, true},
                                           // Exercise local-memory-driven 4 -> 3 -> 2 -> 1 blocking and tails.
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 5000, ov::element::f32, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 3, 6000, ov::element::f32, false},
                                           selective_ssm_test_params{1, 2, 2, 1, 4, 8192, ov::element::f32, false},
                                           // Exceed 128 KiB of FP32 state and exercise the universal global-state fallback.
                                           selective_ssm_test_params{1, 2, 1, 1, 1, large_state_size, ov::element::f32, false, false, true, 1, false, 2e-4f},
                                           selective_ssm_test_params{1, 2, 1, 1, 1, large_state_size, ov::element::f16, true, true}));

}  // namespace
