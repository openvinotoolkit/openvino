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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

// Regression: stride==2 alternating with gate_idx=1 (gate at the odd neighbor, up at the
// even position) — the GPT-OSS pattern that fires from both SwiGluFusionWithClamp and the
// 2-GEMM fused MoE path. The OPT kernel's gate_idx!=0 branch previously reused the split-mode
// formula and read input[y+GLU_STRIDE] for gate; on the last x that reads past the input
// buffer (silently zero on padded shapes, CL_OUT_OF_RESOURCES otherwise). Uses static shape
// with no padding so the OPT kernel (FORCE_PRIORITY_1) is selected — the dynamic-shape
// variant of this test falls back to the REF kernel which has correct logic and would not
// catch the OPT bug.
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

    // Use sequential moderate-magnitude values so that reading input[y+1] (correct, the odd
    // neighbor) vs input[y+2] (buggy, two ahead) produces visibly different swish results —
    // and stays inside the swish "active" region so the bug doesn't get masked by saturation.
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
    // Force the OPT kernel: REF supports tensor offsets/pitches, so the selector picks it
    // by default for these shapes — but the bug we are guarding against lives in the OPT
    // kernel's stride==2 + gate_idx!=0 path, which only fires in production for clean,
    // contiguous tensors (e.g. the gather output in fused MoE).
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