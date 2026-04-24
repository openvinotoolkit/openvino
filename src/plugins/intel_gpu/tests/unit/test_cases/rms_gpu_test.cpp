// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/rms.hpp>
#include "rms_inst.h"

using namespace cldnn;
using namespace ::tests;

class rms_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void rms_ref(const memory::ptr input, const memory::ptr gamma, memory::ptr output, float epsilon) {
    auto input_layout = input->get_layout();

    uint32_t batch_size = input_layout.batch();
    uint32_t feature_size = input_layout.feature();
    uint32_t y_size = input_layout.spatial(1);
    uint32_t x_size = input_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> dst(output, get_test_stream());
    
    std::unique_ptr<cldnn::mem_lock<T>> weight;
    if (gamma) {
        weight = std::make_unique<cldnn::mem_lock<T>>(gamma, get_test_stream());
    }

    // RMS normalization across the last (innermost populated) dimension.
    // For bfyx with x_size > 1 (rank 4): normalize across X per (b, f, y).
    // For bfyx with x_size == 1 (rank 3 mapped to bfyx): normalize across Y per (b, f).
    // This matches the kernel behavior based on ov_input_rank.
    bool norm_over_x = (x_size > 1);
    uint32_t outer_y = norm_over_x ? y_size : 1;
    uint32_t norm_size = norm_over_x ? x_size : y_size;

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t f = 0; f < feature_size; ++f) {
            for (uint32_t oy = 0; oy < outer_y; ++oy) {
                float rms = 0.f;
                for (uint32_t n = 0; n < norm_size; ++n) {
                    uint32_t y = norm_over_x ? oy : n;
                    uint32_t x = norm_over_x ? n : 0;
                    auto t = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    rms += std::pow(static_cast<float>(src[input_layout.get_linear_offset(t)]), 2);
                }
                rms /= norm_size;
                rms += epsilon;
                rms = std::pow(std::sqrt(rms), -1);

                for (uint32_t n = 0; n < norm_size; ++n) {
                    uint32_t y = norm_over_x ? oy : n;
                    uint32_t x = norm_over_x ? n : 0;
                    auto t = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    size_t offset = input_layout.get_linear_offset(t);

                    float gamma_val = weight ? static_cast<float>((*weight)[n]) : 1.0f;
                    dst[offset] = static_cast<T>(rms * static_cast<float>(src[offset]) * gamma_val);
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

TEST(rms_gpu_test, rms_test_bfyx_opt_padding) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{-1, 2, 16}, data_types::f32, format::bfyx, cldnn::padding({0,0,2}, {0,0,4}, 0x4)};

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx, cldnn::padding({0,0,2}, {0,0,4})});
    auto input_ref = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 16}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});

    set_values(input,  {
        0.0f,       0.0f,      0.001839f,  -0.003815f, 0.000961f,  0.002930f, -0.003998f, -0.008057f, -0.005402f, -0.002945f, 0.006744f,
        -0.000004f, 0.004303f, -0.002380f, 0.000072f,  0.001404f,  0.000568f, 0.002579f,  0.0f,       0.0f,       0.0f,       0.0f,
        0.0f,       0.0f,      0.003098f,  -0.006989f, -0.000244f, 0.010193f, 0.002899f,  -0.005798f, -0.026978f, 0.008789f,  0.002258f,
        0.006500f,  0.003159f, -0.012329f, 0.026245f,  -0.001839f, 0.000259f, 0.002670f,  0.0f,       0.0f,       0.0f,       0.0f,
    });

    set_values(input_ref, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f, -0.005402f, -0.002945f,
        0.006744f, -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f, 0.000568f, 0.002579f,
        0.003098f, -0.006989f, -0.000244f, 0.010193f, 0.002899f, -0.005798f, -0.026978f, 0.008789f,
        0.002258f, 0.006500f, 0.003159f, -0.012329f, 0.026245f, -0.001839f, 0.000259f, 0.002670f
    });
    set_values(gamma, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f, 0.008362f, 0.008179f,
        0.018188f, 0.021973f, 0.005249f, 0.004639f, 0.004272f, 0.020264f, 0.013489f, 0.008789f
    });

    rms_ref<float>(input_ref, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        ASSERT_NEAR(output_ptr[i], output_ref_ptr[i], 1e-3) << " index=" << i;
    }
}

TEST(rms_gpu_test, rms_test_without_gamma_bfyx_ref) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});

    set_values(input, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f,
        0.006744f, -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f
    });

    rms_ref<float>(input, nullptr, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(rms("rms", input_info("input"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

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

TEST(rms_gpu_test, rms_test_without_gamma_bfyx_opt) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 16}, data_types::f32, format::bfyx});

    set_values(input, {
        0.001839f, -0.003815f, 0.000961f, 0.002930f, -0.003998f, -0.008057f, -0.005402f, -0.002945f,
        0.006744f, -0.000004f, 0.004303f, -0.002380f, 0.000072f, 0.001404f, 0.000568f, 0.002579f,
        0.003098f, -0.006989f, -0.000244f, 0.010193f, 0.002899f, -0.005798f, -0.026978f, 0.008789f,
        0.002258f, 0.006500f, 0.003159f, -0.012329f, 0.026245f, -0.001839f, 0.000259f, 0.002670f
    });

    rms_ref<float>(input, nullptr, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(rms("rms", input_info("input"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

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

TEST(rms_gpu_test, rms_test_without_gamma_dyn) {
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4096},
                                       data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{2, 1, 4096}, data_types::f32, format::bfyx});

    tests::set_random_values<float>(input, true, 8, 100);

    rms_ref<float>(input, nullptr, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(rms("rms", input_info("input"), 1e-5f));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);

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

// Regression test for in-place crop on spatial axis followed by RMS normalization.
// RMS kernels may not correctly handle spatial padding introduced by in-place crop.
TEST(rms_gpu_test, in_place_crop_rms_spatial_split) {
    auto& engine = get_test_engine();

    // Input [1,2,3,16] bfyx. VariadicSplit on axis 2 (spatial Y) into [2, 1].
    // crop_0 covers y=[0..1] (size 2), crop_1 covers y=2 (size 1).
    // Each crop feeds RMS normalization.
    // dim_x = 16 is the minimum for the bfyx_opt kernel (requires gamma >= subgroup_size(16)).
    // force_implementations ensures bfyx_opt is selected regardless of other heuristics.
    const int64_t dim_b = 1, dim_f = 2, dim_y = 3, dim_x = 16;
    const int64_t split0 = 2, split1 = 1;
    const int64_t off_y0 = 0, off_y1 = 2;
    const float epsilon = 1e-6f;

    auto input_mem = engine.allocate_memory({ov::PartialShape{dim_b, dim_f, dim_y, dim_x},
                                             data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({ov::PartialShape{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({ov::PartialShape{2}, data_types::i64, format::bfyx});
    // Gamma weights for RMS: shape [1, 1, 1, dim_x] — per-element scale on X axis
    auto gamma_mem = engine.allocate_memory({ov::PartialShape{1, 1, 1, dim_x}, data_types::f32, format::bfyx});

    const int64_t axis = 2;
    const size_t total = static_cast<size_t>(dim_b * dim_f * dim_y * dim_x);

    // Deterministic input with small values
    std::vector<float> input_data(total);
    for (size_t i = 0; i < total; i++)
        input_data[i] = static_cast<float>(i + 1);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {split0, split1});
    // Gamma = 1.0 (identity scale) for easy reference computation
    std::vector<float> gamma_data(dim_x, 1.0f);
    set_values(gamma_mem, gamma_data);

    // Prepare crop slices and compute reference outputs using rms_ref
    auto make_crop_ref = [&](int64_t split_size, int64_t y_offset) {
        auto crop_mem = engine.allocate_memory({ov::PartialShape{dim_b, dim_f, split_size, dim_x}, data_types::f32, format::bfyx});
        auto out_mem  = engine.allocate_memory({ov::PartialShape{dim_b, dim_f, split_size, dim_x}, data_types::f32, format::bfyx});
        std::vector<float> crop_data;
        for (int64_t f = 0; f < dim_f; f++)
            for (int64_t y = 0; y < split_size; y++)
                for (int64_t x = 0; x < dim_x; x++)
                    crop_data.push_back(input_data[static_cast<size_t>(f * dim_y * dim_x + (y_offset + y) * dim_x + x)]);
        set_values(crop_mem, crop_data);
        rms_ref<float>(crop_mem, gamma_mem, out_mem, epsilon);
        return out_mem;
    };
    auto output_ref_0 = make_crop_ref(split0, off_y0);
    auto output_ref_1 = make_crop_ref(split1, off_y1);

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(data("axis", axis_mem));
    topology.add(data("splits_length", splits_length_mem));
    topology.add(data("gamma", gamma_mem));
    // Branch 0: crop [1,2,2,16] -> RMS normalization
    topology.add(crop("crop_0", {input_info("input"), input_info("axis"), input_info("splits_length")},
                      tensor(1), tensor(0, 0, 0, off_y0), op_mode, 0, axis));
    topology.add(rms("rms_0", input_info("crop_0"), input_info("gamma"), epsilon));
    topology.add(reorder("output_0", input_info("rms_0"), format::bfyx, data_types::f32,
                         std::vector<float>(), reorder_mean_mode::subtract, padding(), true));
    // Branch 1: crop [1,2,1,16] -> RMS normalization
    topology.add(crop("crop_1", {input_info("input"), input_info("axis"), input_info("splits_length")},
                      tensor(1), tensor(0, 0, 0, off_y1), op_mode, 1, axis));
    topology.add(rms("rms_1", input_info("crop_1"), input_info("gamma"), epsilon));
    topology.add(reorder("output_1", input_info("rms_1"), format::bfyx, data_types::f32,
                         std::vector<float>(), reorder_mean_mode::subtract, padding(), true));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"rms_0", {format::bfyx, "rms_gpu_bfyx_opt"}},
        {"rms_1", {format::bfyx, "rms_gpu_bfyx_opt"}}
    }));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("crop_0")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_1")->can_be_optimized());

    // Verify branch 0: crop [1,2,2,16], y-offset = 0
    auto out0_mem = outputs.at("output_0").get_memory();
    cldnn::mem_lock<float> out0(out0_mem, get_test_stream());
    cldnn::mem_lock<float> ref0(output_ref_0, get_test_stream());
    ASSERT_EQ(out0.size(), ref0.size());
    for (size_t i = 0; i < ref0.size(); i++) {
        ASSERT_NEAR(out0[i], ref0[i], 1e-4f) << "Branch 0 mismatch at index=" << i;
    }

    // Verify branch 1: crop [1,2,1,16], y-offset = 2
    auto out1_mem = outputs.at("output_1").get_memory();
    cldnn::mem_lock<float> out1(out1_mem, get_test_stream());
    cldnn::mem_lock<float> ref1(output_ref_1, get_test_stream());
    ASSERT_EQ(out1.size(), ref1.size());
    for (size_t i = 0; i < ref1.size(); i++) {
        ASSERT_NEAR(out1[i], ref1[i], 1e-4f) << "Branch 1 mismatch at index=" << i;
    }
}
