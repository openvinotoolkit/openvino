// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "feed_forward_inst.h"

using namespace cldnn;
using namespace ::tests;

class feed_forward_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void feed_froward_ref(const memory::ptr input,
                        const memory::ptr input2,
                        const memory::ptr input3,
                        const memory::ptr input4,
                        const memory::ptr input5,
                        memory::ptr output,
                        bool is_scalar = true) {
    auto input_layout = input->get_layout();
    auto output_layout = output->get_layout();

    uint32_t batch_size = output_layout.batch();
    uint32_t feature_size = output_layout.feature();
    uint32_t y_size = output_layout.spatial(1);
    uint32_t x_size = output_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> src2(input2, get_test_stream());
    cldnn::mem_lock<T> src3(input3, get_test_stream());
    cldnn::mem_lock<T> src4(input4, get_test_stream());
    cldnn::mem_lock<T> src5(input5, get_test_stream());
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
                    if (is_scalar)
                        res = (std::tanh(src3[0] * (src2[0] * res * res * res + res)) + src4[0]) * res * src5[0];
                    else
                        res = (std::tanh(src3[src_offset] * (src2[src_offset] * res * res * res + res)) + src4[src_offset]) * res * src5[src_offset];
                    dst[dst_offset] = res;
                }
            }
        }
     }
}


TEST(feed_forward_gpu_test, basic0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f16, format::bfyx});
    auto input2= engine.allocate_memory({ov::PartialShape{1}, data_types::f16, format::bfyx});
    auto input3 = engine.allocate_memory({ov::PartialShape{1}, data_types::f16, format::bfyx});
    auto input4 = engine.allocate_memory({ov::PartialShape{1}, data_types::f16, format::bfyx});
    auto input5 = engine.allocate_memory({ov::PartialShape{1}, data_types::f16, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f16, format::bfyx});

    set_values(input, { ov::float16(-4.5f), ov::float16(-2.5f), ov::float16(-1.5f), ov::float16(0.5f),
                        ov::float16(0.9f),  ov::float16(1.5f),  ov::float16(2.0f),  ov::float16(2.5f),
                        ov::float16(-4.5f), ov::float16(-2.5f), ov::float16(-1.5f), ov::float16(0.5f) });

    set_values(input2, {ov::float16(0.2f)});
    set_values(input3, {ov::float16(0.3f)});
    set_values(input4, {ov::float16(0.4f)});
    set_values(input5, {ov::float16(0.5f)});

    feed_froward_ref<ov::float16>(input, input2, input3, input4, input5, output_ref);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(input_layout("input4", input4->get_layout()));
    topology.add(input_layout("input5", input5->get_layout()));
    topology.add(feed_forward("feed_forward",
                                input_info("input"), 
                                input_info("input2"),
                                input_info("input3"),
                                input_info("input4"),
                                input_info("input5"), tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);
    network.set_input_data("input5", input5);

    auto inst = network.get_primitive("feed_forward");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "feed_forward");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    cldnn::mem_lock<ov::float16> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(feed_forward_gpu_test, basic1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto input2= engine.allocate_memory({ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto input3 = engine.allocate_memory({ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto input4 = engine.allocate_memory({ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto input5 = engine.allocate_memory({ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});

    set_values(input, {
        0.049011f, 0.000260f, -0.176636f, 0.016098f, 0.279297f, 0.036377f,
        -0.127686f, 0.066650f, -0.394043f, -0.135620f, 0.040985f, -0.011589f
    });
    set_values(input2, {0.2f});
    set_values(input3, {0.3f});
    set_values(input4, {0.4f});
    set_values(input5, {0.5f});

    feed_froward_ref<float>(input, input2, input3, input4, input5, output_ref);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(input_layout("input4", input4->get_layout()));
    topology.add(input_layout("input5", input5->get_layout()));
    topology.add(feed_forward("feed_forward",
                                input_info("input"), 
                                input_info("input2"),
                                input_info("input3"),
                                input_info("input4"),
                                input_info("input5"), tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);
    network.set_input_data("input5", input5);

    auto inst = network.get_primitive("feed_forward");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "feed_forward");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(feed_forward_gpu_test, basic2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto input2= engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto input3 = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto input4 = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto input5 = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 6}, data_types::f32, format::bfyx});

    set_values(input, {
        0.049011f, 0.000260f, -0.176636f, 0.016098f, 0.279297f, 0.036377f,
        -0.127686f, 0.066650f, -0.394043f, -0.135620f, 0.040985f, -0.011589f
    });
    set_values(input2, {
        1.049011f, 2.000260f, -1.176636f, 2.016098f, 1.279297f, 2.036377f,
        -0.127686f, 0.066650f, -0.394043f, -0.135620f, 0.040985f, -0.011589f
    });
    set_values(input3, {
        0.049011f, 0.000260f, -0.176636f, 0.016098f, 0.279297f, 0.036377f,
        -2.127686f, 1.066650f, -2.394043f, -1.135620f, 2.040985f, -1.011589f
    });
    set_values(input4, {
        0.049011f, 0.000260f, -0.176636f, 0.016098f, 0.279297f, 0.036377f,
        -9.127686f, 10.066650f, -11.394043f, -12.135620f, 13.040985f, -14.011589f
    });
    set_values(input5, {
        3.049011f, 4.000260f, -5.176636f, 6.016098f, 7.279297f, 8.036377f,
        -0.127686f, 0.066650f, -0.394043f, -0.135620f, 0.040985f, -0.011589f
    });

    feed_froward_ref<float>(input, input2, input3, input4, input5, output_ref, false);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(input_layout("input4", input4->get_layout()));
    topology.add(input_layout("input5", input5->get_layout()));
    topology.add(feed_forward("feed_forward",
                                input_info("input"), 
                                input_info("input2"),
                                input_info("input3"),
                                input_info("input4"),
                                input_info("input5"), tensor()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);
    network.set_input_data("input5", input5);

    auto inst = network.get_primitive("feed_forward");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "feed_forward");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}
