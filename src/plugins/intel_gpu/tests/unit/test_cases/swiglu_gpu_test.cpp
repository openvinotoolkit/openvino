// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include "ov_ops/glu.hpp"
#include "swiglu_inst.h"

using namespace cldnn;
using namespace ::tests;

class swiglu_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void swiglu_ref(const memory::ptr input,
                memory::ptr output,
                int32_t swiglu_axis,
                int32_t gate_idx,
                int32_t glu_stride,
                float clamp_min,
                float clamp_max,
                float swish_beta = 1.0f,
                float up_add_val = 0.0f) {
    auto input_layout = input->get_layout();
    auto output_layout = output->get_layout();

    uint32_t batch_size = output_layout.batch();
    uint32_t feature_size = output_layout.feature();
    uint32_t y_size = output_layout.spatial(1);
    uint32_t x_size = output_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> dst(output, get_test_stream());

    for (uint32_t b = 0; b < batch_size; ++b) {
        auto b_in = b;
        if (glu_stride == 2 && swiglu_axis == 0)
            b_in *= 2;
        for (uint32_t f = 0; f < feature_size; ++f) {
            auto f_in = f;
            if (glu_stride == 2 && swiglu_axis == 1)
                f_in *= 2;
            for (uint32_t y = 0; y < y_size; ++y) {
                auto y_in = y;
                if (glu_stride == 2 && swiglu_axis == 2)
                    y_in *= 2;
                for (uint32_t x = 0; x < x_size; ++x) {
                    auto x_in = x;
                    if (glu_stride == 2 && swiglu_axis == 3)
                        x_in *= 2;
                    size_t src_offset = input_layout.get_linear_offset({static_cast<int32_t>(b_in), static_cast<int32_t>(f_in),
                                                                        static_cast<int32_t>(x_in), static_cast<int32_t>(y_in), 0, 0});
                    size_t dst_offset = output_layout.get_linear_offset({static_cast<int32_t>(b), static_cast<int32_t>(f),
                                                                        static_cast<int32_t>(x), static_cast<int32_t>(y), 0, 0});
                    T gate = src[src_offset];
                    T up = (glu_stride == 2) ? src[src_offset + 1] : src[src_offset + static_cast<size_t>(glu_stride)];
                    if (gate_idx == 1) {
                        std::swap(gate, up);
                    }
                    if (clamp_min != clamp_max) {
                        gate = std::min(static_cast<T>(clamp_max), gate);
                        up = std::min(static_cast<T>(clamp_max), std::max(static_cast<T>(clamp_min), up));
                    }
                    gate = (gate / (static_cast<T>(1) + (std::exp((-(static_cast<T>(swish_beta) * gate))))));
                    T res = (up + up_add_val) * gate;
                    dst[dst_offset] = res;
                }
            }
        }
     }
}

TEST(swiglu_gpu_test, swiglu_test_bfyx_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 6},
                                       data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 3}, data_types::f32, format::bfyx});

    set_values(input_mem, {
        0.049011f, 0.000260f, -0.176636f, 0.016098f, 0.279297f, 0.036377f,
        -0.127686f, 0.066650f, -0.394043f, -0.135620f, 0.040985f, -0.011589f
    });

    swiglu_ref<float>(input_mem, output_ref, 2, 0, 3, 0., 0.);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(swiglu("swiglu", input_info("input"), -1, 3, ov::op::internal::GLU::GluType::Swish, 0, tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto inst = network.get_primitive("swiglu");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(swiglu_gpu_test, swiglu_test_bfyx_dyn_clamp) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 6}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 3}, data_types::f32, format::bfyx});

    auto clamp_min = -0.7;
    auto clamp_max = 7.0;

    set_values(input_mem, {4.9011f, 2.60f, -1.76636f, 0.16098f, 2.79297f, 3.6377f, -0.127686f, 6.6650f, -3.94043f, -1.35620f, 4.0985f, -1.1589f});

    swiglu_ref<float>(input_mem, output_ref, 2, 0, 3, clamp_min, clamp_max);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(swiglu("swiglu", input_info("input"), -1, 3, ov::op::internal::GLU::GluType::Swish, 0, clamp_min, clamp_max, 1.0f, 0.0f, tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto inst = network.get_primitive("swiglu");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(swiglu_gpu_test, swiglu_test_bfyx_dyn_clamp_swish_beta_up_add_val) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 6};
    auto input_layout_dynamic = layout{input_shape, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 3}, data_types::f32, format::bfyx});

    auto clamp_min = -0.7f;
    auto clamp_max = 7.0f;

    int32_t gate_idx = 0;
    int32_t glu_stride = 2;
    float swish_beta = 1.2f;
    float up_add_val = 1.0f;

    set_values(input_mem, {4.9011f, 2.60f, -1.76636f, 0.16098f, 2.79297f, 3.6377f, -0.127686f, 6.6650f, -3.94043f, -1.35620f, 4.0985f, -1.1589f});

    swiglu_ref<float>(input_mem, output_ref, 2, gate_idx, glu_stride, clamp_min, clamp_max, swish_beta, up_add_val);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(swiglu("swiglu", input_info("input"), -1, glu_stride, ov::op::internal::GLU::GluType::Swish, gate_idx, clamp_min, clamp_max, swish_beta, up_add_val, tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto inst = network.get_primitive("swiglu");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

// Regression: stride==2 + gate_idx=1 (GPT-OSS). OPT kernel previously read input[y+GLU_STRIDE]
// for gate instead of input[y+1], causing OOB on the last x. Static shape forces OPT.
TEST(swiglu_gpu_test, swiglu_test_bfyx_static_clamp_swish_beta_up_add_val_gate_idx_1) {
    auto& engine = get_test_engine();
    auto input_layout_static = layout{ov::PartialShape{2, 1, 12}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{2, 1, 12}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});

    auto clamp_min = -0.7f;
    auto clamp_max = 7.0f;

    int32_t gate_idx = 1;
    int32_t glu_stride = 2;
    float swish_beta = 1.2f;
    float up_add_val = 1.0f;

    // Sequential values keep input[y+1] vs input[y+2] swish results distinguishable.
    set_values(input_mem, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f,
    });

    swiglu_ref<float>(input_mem, output_ref, 2, gate_idx, glu_stride, clamp_min, clamp_max, swish_beta, up_add_val);

    topology topology;
    topology.add(input_layout("input", input_layout_static));
    // bfyx tensor for {2, 1, 6} -> b=2, f=1, x=6, y=1 (cldnn::tensor takes b, f, x, y)
    topology.add(swiglu("swiglu", input_info("input"), -1, glu_stride, ov::op::internal::GLU::GluType::Swish, gate_idx, clamp_min, clamp_max, swish_beta, up_add_val, cldnn::tensor(2, 1, 6, 1)));

    ExecutionConfig config = get_test_default_config(engine);
    // Bug lives in OPT kernel; default selector picks REF for these shapes.
    ov::intel_gpu::ImplForcingMap forced{
        {"swiglu", ov::intel_gpu::ImplementationDesc{format::bfyx, "swiglu_gpu_opt"}},
    };
    config.set_property(ov::intel_gpu::force_implementations(forced));

    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

// ============================================================================
// BF16/F16 SwiGLU: reference computed in FP32 mirroring swiglu_gpu_opt/ref kernels.
// Input layout [1, C, W] bfyx (contiguous: index = c*W + x), output [1, C, W_out].
// Supports split mode (glu_stride > 2) and alternating mode (glu_stride == 2),
// gate_idx 0/1 and all three GLU types, plus clamp / swish_beta / up_add_val.
// ============================================================================
static void swiglu_ref_typed(data_types dt, const memory::ptr input, memory::ptr output,
                             int32_t C, int32_t W_out, int32_t glu_stride, int32_t gate_idx,
                             ov::op::internal::GLU::GluType glu_type,
                             float clamp_min, float clamp_max,
                             float swish_beta, float up_add_val) {
    cldnn::mem_lock<ov::bfloat16> src_bf16(input, get_test_stream());
    cldnn::mem_lock<ov::float16> src_f16(input, get_test_stream());
    cldnn::mem_lock<float> dst(output, get_test_stream());
    // Input row holds two halves (gates + values): W = 2 * W_out always.
    const int32_t W = 2 * W_out;

    auto read = [&](size_t i) -> float {
        return dt == data_types::bf16 ? static_cast<float>(src_bf16[i])
                                      : static_cast<float>(src_f16[i]);
    };

    for (int32_t c = 0; c < C; ++c) {
        for (int32_t x = 0; x < W_out; ++x) {
            const size_t base = static_cast<size_t>(c) * W;
            float gate, up;
            if (glu_stride == 2) {
                // alternating: pair is at (base + 2*x, base + 2*x + 1)
                if (gate_idx == 0) {
                    gate = read(base + 2 * x);
                    up = read(base + 2 * x + 1);
                } else {
                    up = read(base + 2 * x);
                    gate = read(base + 2 * x + 1);
                }
            } else {
                // split: input row is blocks of 2*glu_stride: [stride gates | stride values].
                // opt kernel: y = x + (x/stride)*stride => gate at 2*g*stride + i, value at +stride.
                // (classic config glu_stride == W_out degenerates to gate=x, value=x+W_out).
                const int32_t g = x / glu_stride;
                const int32_t i = x % glu_stride;
                const size_t gate_off = base + 2 * g * glu_stride + i;
                if (gate_idx == 0) {
                    gate = read(gate_off);
                    up = read(gate_off + glu_stride);
                } else {
                    up = read(gate_off);
                    gate = read(gate_off + glu_stride);
                }
            }
            if (glu_type == ov::op::internal::GLU::GluType::Swish &&
                (clamp_min > std::numeric_limits<float>::lowest() || clamp_max < std::numeric_limits<float>::max())) {
                gate = std::min(clamp_max, gate);
                up = std::min(clamp_max, std::max(clamp_min, up));
            }
            switch (glu_type) {
            case ov::op::internal::GLU::GluType::Swish:
                gate = gate / (1.0f + std::exp(-swish_beta * gate));
                break;
            case ov::op::internal::GLU::GluType::Gelu:
                gate = 0.5f * gate * (1.0f + std::erf(gate * 0.7071067811865475f));
                break;
            case ov::op::internal::GLU::GluType::Gelu_Tanh:
                gate = 0.5f * gate * (1.0f + std::tanh(0.79788458347320556640625f * gate * (1.0f + 0.044715f * gate * gate)));
                break;
            }
            float res = (up + up_add_val) * gate;
            dst[static_cast<size_t>(c) * W_out + x] = res;
        }
    }
}

// Runs a BF16/F16 swiglu network with forced kernel and compares against the FP32 reference.
static void run_swiglu(data_types dt, const std::string& kernel_name, int32_t C, int32_t W_out, int32_t glu_stride,
                       int32_t gate_idx, ov::op::internal::GLU::GluType glu_type,
                       float clamp_min, float clamp_max, float swish_beta, float up_add_val,
                       bool dynamic = false) {
    auto& engine = get_test_engine();
    // Split mode consumes 2*glu_stride inputs per glu_stride outputs (first half gates, second half
    // values for the classic glu_stride == W_out config); alternating (glu_stride==2) also needs 2*W_out.
    const int32_t W = 2 * W_out;

    auto input_layout_static = layout{ov::PartialShape{1, C, W}, dt, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{1, C, W}, dt, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, C, W_out}, data_types::f32, format::bfyx});

    // Generate random values (7 significand bits for bf16, 10 for f16)
    if (dt == data_types::bf16)
        tests::set_random_values<ov::bfloat16>(input_mem, true, 7, 100);
    else
        tests::set_random_values<ov::float16>(input_mem, true, 10, 100);

    swiglu_ref_typed(dt, input_mem, output_ref, C, W_out, glu_stride, gate_idx, glu_type,
                     clamp_min, clamp_max, swish_beta, up_add_val);

    topology topology;
    topology.add(input_layout("input", input_layout_static));
    topology.add(swiglu("swiglu", input_info("input"), -1, glu_stride,
                        glu_type, gate_idx,
                        clamp_min, clamp_max, swish_beta, up_add_val,
                        cldnn::tensor(1, C, W_out, 1)));

    ExecutionConfig config = get_test_default_config(engine);
    if (dynamic)
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplForcingMap forced{
        {"swiglu", ov::intel_gpu::ImplementationDesc{format::bfyx, kernel_name}},
    };
    config.set_property(ov::intel_gpu::force_implementations(forced));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");
    auto output = outputs.at("swiglu").get_memory();

    cldnn::mem_lock<ov::bfloat16> gpu_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> gpu_f16(output, get_test_stream());
    cldnn::mem_lock<float> ref_ptr(output_ref, get_test_stream());
    auto read_gpu = [&](size_t i) -> float {
        return dt == data_types::bf16 ? static_cast<float>(gpu_bf16[i])
                                      : static_cast<float>(gpu_f16[i]);
    };

    // BF16 ~7 mantissa bits; F16 ~10 but accumulates in half on GPU -> slightly looser
    float abs_floor = (dt == data_types::bf16) ? 0.25f : 0.05f;
    float rel_tol = (dt == data_types::bf16) ? 0.01f : 0.03f;
    for (size_t i = 0; i < output_ref->count(); ++i) {
        float gpu_val = read_gpu(i);
        float ref_val = ref_ptr[i];
        float diff = std::abs(gpu_val - ref_val);
        float tolerance = std::max(abs_floor, std::abs(ref_val) * rel_tol);
        ASSERT_LE(diff, tolerance) << "Mismatch at i=" << i
            << " gpu=" << gpu_val << " ref=" << ref_val << " diff=" << diff;
    }
}

class swiglu_2dtype_test : public ::testing::TestWithParam<data_types> {};

static std::string swiglu_2dtype_test_name(testing::TestParamInfo<data_types> info) {
    return info.param == data_types::bf16 ? "bf16" : "f16";
}

INSTANTIATE_TEST_SUITE_P(smoke, swiglu_2dtype_test,
                         ::testing::Values(data_types::bf16, data_types::f16),
                         swiglu_2dtype_test_name);

// SwiGLU on the ref kernel, split mode, gate_idx=0, Swish
TEST_P(swiglu_2dtype_test, ref_split_swish) {
    run_swiglu(GetParam(), "swiglu_gpu_ref", /*C=*/4, /*W_out=*/512, /*glu_stride=*/512, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the opt kernel, alternating mode (glu_stride==2), gate_idx=0, Swish
TEST_P(swiglu_2dtype_test, opt_alternating) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/4, /*W_out=*/512, /*glu_stride=*/2, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the ref kernel, alternating mode, gate_idx=1 (GPT-OSS style), Swish
TEST_P(swiglu_2dtype_test, ref_alternating_gate_idx_1) {
    run_swiglu(GetParam(), "swiglu_gpu_ref", /*C=*/4, /*W_out=*/512, /*glu_stride=*/2, /*gate_idx=*/1,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the opt kernel, alternating mode, gate_idx=1, Swish
TEST_P(swiglu_2dtype_test, opt_alternating_gate_idx_1) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/4, /*W_out=*/512, /*glu_stride=*/2, /*gate_idx=*/1,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the opt kernel, split mode with block-interleaved stride 8
// (exercises the opt kernel's y = x + (x/stride)*stride indexing)
TEST_P(swiglu_2dtype_test, opt_split_block_stride) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/4, /*W_out=*/512, /*glu_stride=*/8, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the opt kernel, split mode, Gelu_Tanh activation
TEST_P(swiglu_2dtype_test, opt_gelu_tanh) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/4, /*W_out=*/512, /*glu_stride=*/512, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Gelu_Tanh, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the ref kernel, split mode, Gelu activation
TEST_P(swiglu_2dtype_test, ref_gelu) {
    run_swiglu(GetParam(), "swiglu_gpu_ref", /*C=*/4, /*W_out=*/512, /*glu_stride=*/512, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Gelu, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU on the opt kernel, split mode, Swish with clamp + swish_beta + up_add_val
TEST_P(swiglu_2dtype_test, opt_clamp_beta_up_add_val) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/4, /*W_out=*/512, /*glu_stride=*/512, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Swish, -0.7f, 7.0f, 1.2f, 1.0f);
}

// SwiGLU on the ref kernel, split mode, Swish with clamp + swish_beta + up_add_val
TEST_P(swiglu_2dtype_test, ref_clamp_beta_up_add_val) {
    run_swiglu(GetParam(), "swiglu_gpu_ref", /*C=*/4, /*W_out=*/512, /*glu_stride=*/512, /*gate_idx=*/1,
               ov::op::internal::GLU::GluType::Swish, -0.7f, 7.0f, 1.2f, 1.0f);
}

// SwiGLU split mode on the opt kernel, matching the ViT action_expert MLP config
// input [1,10,8192] -> output [1,10,4096]
TEST_P(swiglu_2dtype_test, split_swish_large) {
    run_swiglu(GetParam(), "swiglu_gpu_opt", /*C=*/10, /*W_out=*/4096, /*glu_stride=*/4096, /*gate_idx=*/0,
               ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);
}

// SwiGLU with dynamic shape (auto kernel selection)
TEST_P(swiglu_2dtype_test, dynamic) {
    auto& engine = get_test_engine();
    const auto dt = GetParam();
    const int32_t C = 4, W_out = 512, glu_stride = 512, W = 2 * W_out;

    auto input_layout_dyn = layout{ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   dt, format::bfyx};
    auto input_mem = engine.allocate_memory({ov::PartialShape{1, C, W}, dt, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, C, W_out}, data_types::f32, format::bfyx});

    if (dt == data_types::bf16)
        tests::set_random_values<ov::bfloat16>(input_mem, true, 7, 100);
    else
        tests::set_random_values<ov::float16>(input_mem, true, 10, 100);
    swiglu_ref_typed(dt, input_mem, output_ref, C, W_out, glu_stride, 0,
                     ov::op::internal::GLU::GluType::Swish, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f);

    topology topology;
    topology.add(input_layout("input", input_layout_dyn));
    topology.add(swiglu("swiglu", input_info("input"), -1, glu_stride,
                        ov::op::internal::GLU::GluType::Swish, 0,
                        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 1.0f, 0.0f, cldnn::tensor(1, C, W_out, 1)));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto inst = network.get_primitive("swiglu");
    ASSERT_TRUE(inst->get_impl() != nullptr);
    ASSERT_TRUE(inst->get_impl()->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "swiglu");
    auto output = outputs.at("swiglu").get_memory();

    cldnn::mem_lock<ov::bfloat16> gpu_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> gpu_f16(output, get_test_stream());
    cldnn::mem_lock<float> ref_ptr(output_ref, get_test_stream());
    auto read_gpu = [&](size_t i) -> float {
        return dt == data_types::bf16 ? static_cast<float>(gpu_bf16[i])
                                      : static_cast<float>(gpu_f16[i]);
    };

    float abs_floor = (dt == data_types::bf16) ? 0.25f : 0.05f;
    float rel_tol = (dt == data_types::bf16) ? 0.01f : 0.03f;
    for (size_t i = 0; i < output_ref->count(); ++i) {
        float gpu_val = read_gpu(i);
        float ref_val = ref_ptr[i];
        float diff = std::abs(gpu_val - ref_val);
        float tolerance = std::max(abs_floor, std::abs(ref_val) * rel_tol);
        ASSERT_LE(diff, tolerance) << "Mismatch at i=" << i
            << " gpu=" << gpu_val << " ref=" << ref_val << " diff=" << diff;
    }
}
