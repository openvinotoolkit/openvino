// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include "swiglu_inst.h"

using namespace cldnn;
using namespace ::tests;

class swiglu_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void swiglu_ref(const memory::ptr input, memory::ptr output, int32_t split_length) {
    auto input_layout = input->get_layout();
    auto output_layout = output->get_layout();

    uint32_t batch_size = output_layout.batch();
    uint32_t feature_size = output_layout.feature();
    uint32_t y_size = output_layout.spatial(1);
    uint32_t x_size = output_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> dst(output, get_test_stream());

    T res;
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t f = 0; f < feature_size; ++f) {
            for (uint32_t y = 0; y < y_size; ++y) {
                for (uint32_t x = 0; x < x_size; ++x) {
                    size_t src_offset = input_layout.get_linear_offset({static_cast<int32_t>(b), static_cast<int32_t>(f),
                                                                        static_cast<int32_t>(x), static_cast<int32_t>(y), 0, 0});
                    size_t dst_offset = output_layout.get_linear_offset({static_cast<int32_t>(b), static_cast<int32_t>(f),
                                                                        static_cast<int32_t>(x), static_cast<int32_t>(y), 0, 0});
                    res = src[src_offset];
                    res = (res / (static_cast<T>(1) + (std::exp((-(static_cast<T>(1) * res))))));
                    res *= src[src_offset + static_cast<size_t>(split_length)];
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

    swiglu_ref<float>(input_mem, output_ref, 3);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(swiglu("swiglu", input_info("input"), -1, 3, ov::intel_gpu::op::SwiGLU::GluType::Swish, 0, tensor()));

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
