// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/rms.hpp>
#include "rms_inst.h"

using namespace cldnn;
using namespace ::tests;

class rms_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void rms_ref(const memory::ptr input, const memory::ptr gamma, memory::ptr output, float epsilon) {
    auto input_layout = input->get_layout();
    auto gamma_layout = gamma->get_layout();

    uint32_t batch_size = input_layout.batch();
    uint32_t feature_size = input_layout.feature();
    uint32_t y_size = input_layout.spatial(1);
    uint32_t x_size = input_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> weight(gamma, get_test_stream());
    cldnn::mem_lock<T> dst(output, get_test_stream());

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t f = 0; f < feature_size; ++f) {
            float rms = 0.f;
            for (uint32_t y = 0; y < y_size; ++y) {
                for (uint32_t x = 0; x < x_size; ++x) {
                    auto tensor_src = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    size_t src_offset = input_layout.get_linear_offset(tensor_src);
                    rms += std::pow(static_cast<float>(src[src_offset]), 2);
                }
            }
            rms /= y_size * x_size;
            rms += epsilon;
            rms = std::pow(std::sqrt(rms), -1);

            for (uint32_t y = 0; y < y_size; ++y) {
                for (uint32_t x = 0; x < x_size; ++x) {
                    auto tensor_src = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    auto tensor_weight = tensor(batch(0), feature(0), spatial(x, y, 0, 0));
                    auto tensor_dst = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    size_t src_offset = input_layout.get_linear_offset(tensor_src);
                    size_t weight_offset = input_layout.get_linear_offset(tensor_weight);
                    size_t dst_offset = input_layout.get_linear_offset(tensor_dst);
                    float result = rms * static_cast<float>(src[src_offset]) * static_cast<float>(weight[weight_offset]);
                    dst[dst_offset] = static_cast<T>(result);
                }
            }
        }
    }
}

TEST(rms_gpu_test, rms_test_bfyx_ref) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});

    set_values(input, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f,
        0.006744f, -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f
    });
    set_values(gamma, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f
    });

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 16}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});

    set_values(input, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f, -0.005402f, -0.002945f,
        0.006744f, -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f, 0.000568f, 0.002579f,
        0.003098f, -0.006989f, -0.000244f, 0.010193f, 0.002899f, -0.005798f, -0.026978f, 0.008789f,
        0.002258f, 0.006500f, 0.003159f, -0.012329f, 0.026245f, -0.001839f, 0.000259f, 0.002670f
    });
    set_values(gamma, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f, 0.008362f, 0.008179f,
        0.018188f, 0.021973f, 0.005249f, 0.004639f, 0.004272f, 0.020264f, 0.013489f, 0.008789f
    });

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt_leftovers) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 18}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 18}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 18}, data_types::f32, format::bfyx});

    set_values(input, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f, -0.005402f, -0.002945f, 0.006744f,
        -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f, 0.000568f, 0.002579f, 0.003098f, -0.006989f,
        -0.000244f, 0.010193f, 0.002899f, -0.005798f, -0.026978f, 0.008789f, 0.002258f, 0.006500f, 0.003159f,
        -0.012329f, 0.026245f, -0.001839f, 0.000259f, 0.002670f, 0.001419f, 0.001617f,-0.006622f, 0.010864f
    });
    set_values(gamma, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f, 0.008362f, 0.008179f, 0.018188f,
        0.021973f, 0.005249f, 0.004639f, 0.004272f, 0.020264f, 0.013489f, 0.008789f, 0.006653f, 0.010315f
    });

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4096},
                                       data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 1, 4096}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});

    tests::set_random_values<float>(input, true, 8, 100);
    tests::set_random_values<float>(gamma, true, 8, 100);

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto inst = network.get_primitive("rms");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt_all_dims_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 1, 4096}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});

    tests::set_random_values<float>(input, true, 8, 100);
    tests::set_random_values<float>(gamma, true, 8, 100);

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto inst = network.get_primitive("rms");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt_leftovers_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 18},
                                       data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 18}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 1, 18}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 18}, data_types::f32, format::bfyx});

    tests::set_random_values<float>(input, true, 8, 100);
    tests::set_random_values<float>(gamma, true, 8, 100);

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto inst = network.get_primitive("rms");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}

TEST(rms_gpu_test, rms_test_bfyx_opt_unaligned_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3083},
                                       data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 3083}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 1, 3083}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 3083}, data_types::f32, format::bfyx});

    tests::set_random_values<float>(input, true, 8, 100);
    tests::set_random_values<float>(gamma, true, 8, 100);

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto inst = network.get_primitive("rms");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3);
    }
}
