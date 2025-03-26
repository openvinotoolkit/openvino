// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "concatenation_inst.h"
#include "permute_inst.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/grid_sample.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>

using namespace cldnn;
using namespace ::tests;

TEST(concat_gpu, mixed_input_types) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 4, 3 } });
    auto input4 = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 4, 3 } });

    set_values<float>(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values<int32_t>(input1, { 11, 12, 13, 14, 12, 12, 13, 14, 13, 13, 13, 15 });
    set_values<int8_t>(input2, { 21, 22, 23, 24, 22, 22, 23, 24, 23, 23, 23, 25 });
    set_values(input3, { ov::float16(31.f), ov::float16(32.f), ov::float16(33.f),
                         ov::float16(34.f), ov::float16(32.f), ov::float16(32.f),
                         ov::float16(33.f), ov::float16(34.f), ov::float16(33.f),
                         ov::float16(33.f), ov::float16(33.f), ov::float16(35.f) });
    set_values<int64_t>(input4, { 41, 42, 43, 44, 42, 42, 43, 44, 43, 43, 43, 45 });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            input_layout("input4", input4->get_layout()),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3"), input_info("input4") },
                          1,
                          data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 5);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_cpu, disable_usm) {
    auto engine = create_test_engine(engine_types::ocl, runtime_types::ocl, false);

    auto input0 = engine->allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = engine->allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = engine->allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = engine->allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input4 = engine->allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });

    set_values<int8_t>(input0, { 1, 2, 3, 4, 2, 2, 3, 4, 3, 3, 3, 5 });
    set_values<int8_t>(input1, { 11, 12, 13, 14, 12, 12, 13, 14, 13, 13, 13, 15 });
    set_values<int8_t>(input2, { 21, 22, 23, 24, 22, 22, 23, 24, 23, 23, 23, 25 });
    set_values<int8_t>(input3, { 31, 32, 33, 34, 32, 32, 33, 34, 33, 33, 33, 35 });
    set_values<int8_t>(input4, { 41, 42, 43, 44, 42, 42, 43, 44, 43, 43, 43, 45 });

    std::vector<int8_t> output_vec = {
            1, 2, 3, 4, 2, 2, 3, 4, 3, 3, 3, 5,
            11, 12, 13, 14, 12, 12, 13, 14, 13, 13, 13, 15,
            21, 22, 23, 24, 22, 22, 23, 24, 23, 23, 23, 25,
            31, 32, 33, 34, 32, 32, 33, 34, 33, 33, 33, 35,
            41, 42, 43, 44, 42, 42, 43, 44, 43, 43, 43, 45 };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            input_layout("input4", input4->get_layout()),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3"), input_info("input4") },
                          1,
                          data_types::i8)
    );
    ExecutionConfig cfg = get_test_default_config(*engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"concat", {format::bfyx, "", impl_types::cpu}} }));
    network network(*engine, topology, cfg);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 5);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

void start_concat_test_dynamic(impl_types impl_type = impl_types::any);
void start_concat_test_dynamic(impl_types impl_type) {
    auto& engine = get_test_engine();

    layout layout0_dyn = {{1, -1, -1, -1}, data_types::f32, format::bfyx};
    layout layout1_dyn = {{1, -1,  3, -1}, data_types::f32, format::bfyx};
    layout layout2_dyn = {{1,  3,  3, -1}, data_types::f32, format::bfyx};
    layout layout3_dyn = {{1, -1, -1, -1}, data_types::f32, format::bfyx};

    topology topology(
            input_layout("input0", layout0_dyn),
            input_layout("input1", layout1_dyn),
            input_layout("input2", layout2_dyn),
            input_layout("input3", layout3_dyn),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3") },
                          1,
                          data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (impl_type != impl_types::any) {
        auto force_impl = ov::intel_gpu::ImplementationDesc{ format::bfyx, "", impl_type };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {primitive_id("concat"), force_impl} }));
    }

    auto network = cldnn::network::build_network(engine, topology, config);

    auto run_on_shapes = [&](layout layout0, layout layout1, layout layout2, layout layout3) {
        auto input0 = engine.allocate_memory(layout0);
        auto input1 = engine.allocate_memory(layout1);
        auto input2 = engine.allocate_memory(layout2);
        auto input3 = engine.allocate_memory(layout3);

        int counter = 0;

        {
            cldnn::mem_lock<float> ptr0(input0, get_test_stream());
            cldnn::mem_lock<float> ptr1(input1, get_test_stream());
            cldnn::mem_lock<float> ptr2(input2, get_test_stream());
            cldnn::mem_lock<float> ptr3(input3, get_test_stream());

            for (size_t i = 0; i < input0->count(); i++) {
                ptr0[i] = counter++;
            }
            for (size_t i = 0; i < input1->count(); i++) {
                ptr1[i] = counter++;
            }
            for (size_t i = 0; i < input2->count(); i++) {
                ptr2[i] = counter++;
            }
            for (size_t i = 0; i < input3->count(); i++) {
                ptr3[i] = counter++;
            }
        }
        std::vector<float> expected_out(input0->count() + input1->count() + input2->count() + input3->count());
        std::iota(std::begin(expected_out), std::end(expected_out), 0);

        network->set_input_data("input0", input0);
        network->set_input_data("input1", input1);
        network->set_input_data("input2", input2);
        network->set_input_data("input3", input3);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "concat");

        auto output_memory = outputs.at("concat").get_memory();
        auto output_layout = outputs.at("concat").get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        ov::PartialShape expected_shape = layout0.get_partial_shape();
        expected_shape[1] = layout0.get_partial_shape()[1] +
                            layout1.get_partial_shape()[1] +
                            layout2.get_partial_shape()[1] +
                            layout3.get_partial_shape()[1];

        ASSERT_EQ(output_layout.get_partial_shape(), expected_shape);

        for (size_t i = 0; i < output_layout.count(); ++i) {
            ASSERT_EQ(expected_out[i], output_ptr[i]) << " i = " << i;
        }
    };


    run_on_shapes({{1, 3, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 5, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 3, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 1, 3, 2}, data_types::f32, format::bfyx});

    run_on_shapes({{1, 2, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 5, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 3, 3, 2}, data_types::f32, format::bfyx},
                  {{1, 2, 3, 2}, data_types::f32, format::bfyx});

    run_on_shapes({{1, 2, 3, 4}, data_types::f32, format::bfyx},
                  {{1, 5, 3, 4}, data_types::f32, format::bfyx},
                  {{1, 3, 3, 4}, data_types::f32, format::bfyx},
                  {{1, 2, 3, 4}, data_types::f32, format::bfyx});

    if (impl_type == impl_types::cpu) {
        run_on_shapes({{1, 2, 3, 4}, data_types::f32, format::bfyx},
                    {{1, 0, 3, 4}, data_types::f32, format::bfyx},
                    {{1, 3, 3, 4}, data_types::f32, format::bfyx},
                    {{1, 8, 3, 4}, data_types::f32, format::bfyx});
    }
}

TEST(concat_gpu, dynamic_4d_f) {
    start_concat_test_dynamic();
}

TEST(concat_cpu_impl, dynamic_4d_f) {
    start_concat_test_dynamic(impl_types::cpu);
}

TEST(concat_gpu, dynamic_2d_bfyx_and_b_fs_yx_fsv32) {
    auto& engine = get_test_engine();

    topology topology(
            input_layout("input0", { {  2, 4 }, data_types::f32, format::bfyx }),
            input_layout("input1", { { -1, 1 }, data_types::f32, format::bfyx }),
            reorder("reorder_input1", input_info("input1"), { { -1, 1 }, data_types::f16, format::b_fs_yx_fsv32 }),
            concatenation("concat",
                          { input_info("input0"), input_info("reorder_input1") },
                          1,
                          data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc impl = { format::bfyx, "", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "concat", impl } }));

    auto network = cldnn::network::build_network(engine, topology, config);

    layout layout0 = { { 2, 4 }, data_types::f32, format::bfyx };
    layout layout1 = { { 2, 1 }, data_types::f32, format::bfyx };

    auto input0 = engine.allocate_memory(layout0);
    auto input1 = engine.allocate_memory(layout1);

    set_values<float>(input0, { 0, 1, 2, 3, 4, 5, 6, 7 });
    set_values<float>(input1, { 8, 9 });
    VF<float> expected_out = { 0, 1, 2, 3, 8, 4, 5, 6, 7, 9 };

    network->set_input_data("input0", input0);
    network->set_input_data("input1", input1);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = outputs.at("concat").get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    ov::PartialShape expected_shape = layout0.get_partial_shape();
    expected_shape[1] = layout0.get_partial_shape()[1] +
                        layout1.get_partial_shape()[1];

    ASSERT_EQ(output_layout.get_partial_shape(), expected_shape);

    for (size_t i = 0; i < output_layout.count(); ++i) {
        ASSERT_EQ(expected_out[i], output_ptr[i]) << " i = " << i;
    }
}

TEST(concat_gpu, dynamic_4d_bfyx_and_b_fs_yx_fsv32) {
    auto& engine = get_test_engine();

    topology topology(
            input_layout("input0", { { -1, -1, -1, -1 }, data_types::f32, format::bfyx }),
            input_layout("input1", { { -1, -1, -1, -1 }, data_types::f32, format::bfyx }),
            reorder("reorder_input1", input_info("input1"), { { -1, -1, -1, -1 }, data_types::f16, format::b_fs_yx_fsv32 }),
            concatenation("concat",
                          { input_info("input0"), input_info("reorder_input1") },
                          1,
                          data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc impl = { format::bfyx, "", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "concat", impl } }));

    auto network = cldnn::network::build_network(engine, topology, config);

    layout layout0 = { { 2, 4, 1, 1 }, data_types::f32, format::bfyx };
    layout layout1 = { { 2, 1, 1, 1 }, data_types::f32, format::bfyx };

    auto input0 = engine.allocate_memory(layout0);
    auto input1 = engine.allocate_memory(layout1);

    set_values<float>(input0, { 0, 1, 2, 3, 4, 5, 6, 7 });
    set_values<float>(input1, { 8, 9 });
    VF<float> expected_out = { 0, 1, 2, 3, 8, 4, 5, 6, 7, 9 };

    network->set_input_data("input0", input0);
    network->set_input_data("input1", input1);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = outputs.at("concat").get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    ov::PartialShape expected_shape = layout0.get_partial_shape();
    expected_shape[1] = layout0.get_partial_shape()[1] +
                        layout1.get_partial_shape()[1];

    ASSERT_EQ(output_layout.get_partial_shape(), expected_shape);

    for (size_t i = 0; i < output_layout.count(); ++i) {
        ASSERT_EQ(expected_out[i], output_ptr[i]) << " i = " << i;
    }
}

TEST(concat_gpu, dynamic_6d_f) {
    auto& engine = get_test_engine();

    layout layout0_dyn = {{1, -1, -1, -1, -1, -1}, data_types::f32, format::bfwzyx};
    layout layout1_dyn = {{1, -1,  3, -1, -1, -1}, data_types::f32, format::bfwzyx};
    layout layout2_dyn = {{1,  3,  3, -1, -1, -1}, data_types::f32, format::bfwzyx};
    layout layout3_dyn = {{1, -1, -1, -1, -1, -1}, data_types::f32, format::bfwzyx};

    topology topology(
            input_layout("input0", layout0_dyn),
            input_layout("input1", layout1_dyn),
            input_layout("input2", layout2_dyn),
            input_layout("input3", layout3_dyn),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3") },
                          1,
                          data_types::f32)
    );

    ExecutionConfig config{ov::intel_gpu::allow_new_shape_infer(true)};

    auto network = cldnn::network::build_network(engine, topology, config);

    auto run_on_shapes = [&](layout layout0, layout layout1, layout layout2, layout layout3) {
        auto input0 = engine.allocate_memory(layout0);
        auto input1 = engine.allocate_memory(layout1);
        auto input2 = engine.allocate_memory(layout2);
        auto input3 = engine.allocate_memory(layout3);

        int counter = 0;

        {
            cldnn::mem_lock<float> ptr0(input0, get_test_stream());
            cldnn::mem_lock<float> ptr1(input1, get_test_stream());
            cldnn::mem_lock<float> ptr2(input2, get_test_stream());
            cldnn::mem_lock<float> ptr3(input3, get_test_stream());

            for (size_t i = 0; i < input0->count(); i++) {
                ptr0[i] = counter++;
            }
            for (size_t i = 0; i < input1->count(); i++) {
                ptr1[i] = counter++;
            }
            for (size_t i = 0; i < input2->count(); i++) {
                ptr2[i] = counter++;
            }
            for (size_t i = 0; i < input3->count(); i++) {
                ptr3[i] = counter++;
            }
        }
        std::vector<float> expected_out(input0->count() + input1->count() + input2->count() + input3->count());
        std::iota(std::begin(expected_out), std::end(expected_out), 0);

        network->set_input_data("input0", input0);
        network->set_input_data("input1", input1);
        network->set_input_data("input2", input2);
        network->set_input_data("input3", input3);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "concat");

        auto output_memory = outputs.at("concat").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        ov::PartialShape expected_shape = layout0.get_partial_shape();
        expected_shape[1] = layout0.get_partial_shape()[1] +
                            layout1.get_partial_shape()[1] +
                            layout2.get_partial_shape()[1] +
                            layout3.get_partial_shape()[1];

        ASSERT_EQ(output_layout.get_partial_shape(), expected_shape);

        for (size_t i = 0; i < output_layout.count(); ++i) {
            ASSERT_EQ(expected_out[i], output_ptr[i]) << " i = " << i;
        }
    };


    run_on_shapes({{1, 3, 3, 2, 3, 4}, data_types::f32, format::bfwzyx},
                  {{1, 5, 3, 2, 3, 4}, data_types::f32, format::bfwzyx},
                  {{1, 3, 3, 2, 3, 4}, data_types::f32, format::bfwzyx},
                  {{1, 1, 3, 2, 3, 4}, data_types::f32, format::bfwzyx});

    run_on_shapes({{1, 2, 3, 2, 2, 2}, data_types::f32, format::bfwzyx},
                  {{1, 5, 3, 2, 2, 2}, data_types::f32, format::bfwzyx},
                  {{1, 3, 3, 2, 2, 2}, data_types::f32, format::bfwzyx},
                  {{1, 2, 3, 2, 2, 2}, data_types::f32, format::bfwzyx});

    run_on_shapes({{1, 2, 3, 4, 1, 3}, data_types::f32, format::bfwzyx},
                  {{1, 5, 3, 4, 1, 3}, data_types::f32, format::bfwzyx},
                  {{1, 3, 3, 4, 1, 3}, data_types::f32, format::bfwzyx},
                  {{1, 2, 3, 4, 1, 3}, data_types::f32, format::bfwzyx});
}

TEST(concat_gpu, mixed_input_types_5d) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });

    set_values(input0, { ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
                         ov::float16(4.0f), ov::float16(2.0f), ov::float16(2.0f),
                         ov::float16(3.0f), ov::float16(4.0f), ov::float16(3.0f),
                         ov::float16(3.0f), ov::float16(3.0f), ov::float16(5.0f) });
    set_values(input1, { ov::float16(11), ov::float16(12), ov::float16(13),
                         ov::float16(14), ov::float16(12), ov::float16(12),
                         ov::float16(13), ov::float16(14), ov::float16(13),
                         ov::float16(13), ov::float16(13), ov::float16(15) });
    set_values(input2, { ov::float16(21), ov::float16(22), ov::float16(23),
                         ov::float16(24), ov::float16(22), ov::float16(22),
                         ov::float16(23), ov::float16(24), ov::float16(23),
                         ov::float16(23), ov::float16(23), ov::float16(25) });
    set_values(input3, { ov::float16(31.f), ov::float16(32.f), ov::float16(33.f),
                         ov::float16(34.f), ov::float16(32.f), ov::float16(32.f),
                         ov::float16(33.f), ov::float16(34.f), ov::float16(33.f),
                         ov::float16(33.f), ov::float16(33.f), ov::float16(35.f) });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3") },
                          1,
                          data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 1);
    ASSERT_EQ(f_size, 4);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}


TEST(concat_gpu, pooling_dynamic_input_no_exception) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 8, 3}});
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 8, 3}});

    auto input_dyn_layout = layout{ ov::PartialShape{ 1, ov::Dimension(), 8, 2 }, data_types::f32, format::bfyx };
    auto input_actual_grid = layout{ ov::PartialShape{ 1, 3, 8, 2 }, data_types::f32, format::bfyx};
    auto input_grid = engine.allocate_memory(input_actual_grid);

    set_values(input_grid, { 13, 13, 13, 13, 15, 15,
                        16, 15, 16, 14, 13, 14,
                        13, 14, 13, 18, 16, 18,
                        16, 15, 16, 15, 18, 14 });

    set_values<float>(input0, { 11, 12, 13,
                                 14, 12, 12,
                                 13, -14, 13,
                                 13, -13, 15,
                                 16, -16, -13,
                                 -14, 12, 11,
                                 16, -14, -13,
                                 18, -13, -15, });
    set_values<float>(input1, { 11, 12, 13,
                         15, 12, 12,
                         13, 14, 12,
                         13, 13, 15,
                         12, 14, 13,
                         14, 17, 18,
                         13, 14, 11,
                         13, 13, 15 });

    GridSampleOp::Attributes attributes(false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS);

    layout reorder_layout(data_types::f32, format::yxfb, {7, 2, 2, 1});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      input_layout("input_dyn", input_dyn_layout),
                      grid_sample("grid_sample", { input_info("input0"), input_info("input_dyn") }, attributes),
                      pooling("pool0", input_info("grid_sample"), pooling_mode::max, {2, 2}, {1, 1}),
                      pooling("pool1", input_info("input1"), pooling_mode::max, {2, 2}, {1, 1}),
                      concatenation("concat",
                                    { input_info("pool0"), input_info("pool1") },
                                    1,
                                    data_types::f32),
                      reorder("reorder", input_info("concat"), reorder_layout));
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input_dyn", input_grid);

    EXPECT_NO_THROW(network.execute());
}

TEST(concat_gpu, i8_optimization_with_pool) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 8, 3}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 8, 3}});


    set_values<int8_t>(input0, { 11, 12, 13,
                         14, 12, 12,
                         13, -14, 13,
                         13, -13, 15,
                         16, -16, -13,
                         -14, 12, 11,
                         16, -14, -13,
                         18, -13, -15, });
    set_values<int8_t>(input1, { 11, 12, 13,
                         15, 12, 12,
                         13, 14, 12,
                         13, 13, 15,
                         12, 14, 13,
                         14, 17, 18,
                         13, 14, 11,
                         13, 13, 15 });


    VF<int8_t> output_vec = {13, 13, 13, 13, 15, 15,
                        16, 15, 16, 14, 13, 14,
                        13, 14, 13, 18, 16, 18,
                        16, 15, 16, 15, 18, 14,
                        18, 14, -13, 15};

    layout reorder_layout(data_types::i8, format::yxfb, {7, 2, 2, 1});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      pooling("pool0", input_info("input0"), pooling_mode::max, {2, 2}, {1, 1}),
                      pooling("pool1", input_info("input1"), pooling_mode::max, {2, 2}, {1, 1}),
                      concatenation("concat",
                                    { input_info("pool0"), input_info("pool1") },
                                    1,
                                    data_types::i8),
                      reorder("reorder", input_info("concat"), reorder_layout));
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output_memory = outputs.at("reorder").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(0);
    int x_size = output_layout.spatial(1);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 7);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, i8_optimization_with_conv) {
    //  Filter : 3x2x3
    //  Stride : 2x1
    //  Input1  : 4x5
    //  Input2  : 4x5
    //  Input3  : 4x5
    //  Concat output  : 3x4x5
    //  Conv input  : 3x4x5
    //  Output : 2x3
    //
    //  Input0:
    //  1  2  3  -4  5
    //  2  2  3  4  -6
    //  -3  3  3  5  1
    //  -1  1  1  1  -1
    //  Input1:
    //  5  5  3  -4  5
    //  2  -2  5  4  6
    //  6  1  3  5  1
    //  1  2  -3  -4  5
    //  Input2:
    //  -2  1  3  2  -5
    //  1  2  -2  4  2
    //  3  5  3  -3  1
    //  5  4  3  2  1
    //
    //  Filter:
    //  1  2  1     1  2  1     1  2  1
    //  2  1  2     2  1  2     2  1  2
    //
    //  Output:
    // 53  54  30
    // 52  47  37
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto input2 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 3, 3, 2 } });

    set_values<int8_t>(weights, { 1, 2, 1,
                          2, 1, 2, 1, 2, 1,
                          2, 1, 2, 1, 2, 1,
                          2, 1, 2 });

    set_values<int8_t>(input0, {  1, 2, 3, -4, 5,
                          2, 2, 3, 4, -6,
                          -3, 3, 3, 5, 1,
                          -1, 1, 1, 1, -1 });
    set_values<int8_t>(input1, { 5, 5, 3, -4, 5,
                         2, -2, 5, 4, 6,
                         6, 1, 3, 5, 1,
                         1, 2, -3, -4, 5 });
    set_values<int8_t>(input2, {  -2, 1, 3, 2, -5,
                          1, 2, -2, 4, 2,
                          3, 5, 3, -3, 1,
                          5, 4, 3, 2, 1 });

    VF<int8_t> output_vec = { 53, 54, 30, 52, 47, 37 };


    layout reorder_layout(data_types::i8, format::bfyx, {1, 1, 2, 3});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      input_layout("input2", input2->get_layout()),
                      concatenation("concat",
                                    { input_info("input0"), input_info("input1"), input_info("input2") },
                                    1,
                                    data_types::i8),
                      data("weights", weights),
                      convolution("conv", input_info("concat"), "weights", "", 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false),
                      reorder("output", input_info("conv"), reorder_layout));
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, i8_optimization_with_pool_conv) {
    //  Filter : 32x2x1
    //  Input offset : 0x0x-1x0
    //  Stride : 1x1
    //  Input0  : 16x3x2
    //  Input1  : 16x3x2
    //  Output : 1x1x3
    //
    //  Input0:
    // -3 6 0 2 -1 -1 6 0 5 4 1 6 2 4 0 5
    // -2 -1 1 0 2 3 3 3 6 2 4 7 3 6 7 -1
    // 7 7 5 -3 1 -1 5 4 0 3 -2 6 2 5 2 4
    // 5 -1 3 6 2 0 -3 -1 0 3 0 -1 1 6 1 6
    // 5 -2 2 -1 5 6 3 4 1 0 6 6 7 2 6 3
    // 6 7 -1 5 5 6 -1 0 -1 5 5 2 3 -1 -3 4
    //
    //  Input1:
    //  4 -2 0 0 6 2 0 4 6 4 4 4 -3 -1 4 -3
    //  1 0 -1 5 -1 1 4 2 7 7 0 2 3 4 -1 3
    //  7 7 2 -3 -1 5 -2 2 6 -3 0 7 0 3 3 3
    //  -1 0 -2 -2 7 -3 -3 -1 5 0 3 4 0 -1 2 5
    //  2 -1 2 -3 0 -3 -3 2 4 3 3 5 5 7 5 1
    //  2 2 -3 6 6 7 1 -1 -2 5 1 -1 4 5 -3 -2
    //
    // Filters:
    // -1, 2, -2, 2, -2, 1, 1, 0, -1, 1, 2, -2, 2, 1, -2, 0,
    // 0, -2, -2, -2, -2, -1, 2, 1, 2, -1, -1, 0, 2, -2, -2, 1,
    // 0, -2, 0, 1, -2, -1, -2, 0, -1, -1, -2, 1, -2, 0, 1, 2,
    // 2, 2, 2, -2, 0, 2, 1, -2, -1, -1, 0, -2, 2, -1, 2, -1
    //
    //  Output:
    //  -14, -35, -10

    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 16, 3, 2}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 16, 3, 2}});
    auto weights = engine.allocate_memory({data_types::i8, format::bfyx, {1, 32, 2, 1}});

    set_values<int8_t>(weights, {-1, 2, -2, 2, -2, 1, 1, 0, -1, 1, 2, -2, 2, 1, -2, 0, 0, -2, -2, -2, -2, -1, 2, 1, 2, -1, -1, 0, 2, -2, -2, 1,
                                0, -2, 0, 1, -2, -1, -2, 0, -1, -1, -2, 1, -2, 0, 1, 2, 2, 2, 2, -2, 0, 2, 1, -2, -1, -1, 0, -2, 2, -1, 2, -1});

    set_values<int8_t>(input0, {-3, 6, 0, 2, -1, -1, 6, 0, 5, 4, 1, 6, 2, 4, 0, 5,
                                -2, -1, 1, 0, 2, 3, 3, 3, 6, 2, 4, 7, 3, 6, 7, -1,
                                7, 7, 5, -3, 1, -1, 5, 4, 0, 3, -2, 6, 2, 5, 2, 4,
                                5, -1, 3, 6, 2, 0, -3, -1, 0, 3, 0, -1, 1, 6, 1, 6,
                                5, -2, 2, -1, 5, 6, 3, 4, 1, 0, 6, 6, 7, 2, 6, 3,
                                6, 7, -1, 5, 5, 6, -1, 0, -1, 5, 5, 2, 3, -1, -3, 4 });

    set_values<int8_t>(input1, { 4, -2, 0, 0, 6, 2, 0, 4, 6, 4, 4, 4, -3, -1, 4, -3,
                                 1, 0, -1, 5, -1, 1, 4, 2, 7, 7, 0, 2, 3, 4, -1, 3,
                                 7, 7, 2, -3, -1, 5, -2, 2, 6, -3, 0, 7, 0, 3, 3, 3,
                                 -1, 0, -2, -2, 7, -3, -3, -1, 5, 0, 3, 4, 0, -1, 2, 5,
                                 2, -1, 2, -3, 0, -3, -3, 2, 4, 3, 3, 5, 5, 7, 5, 1,
                                 2, 2, -3, 6, 6, 7, 1, -1, -2, 5, 1, -1, 4, 5, -3, -2});

    VF<int8_t> output_vec = { -14, -35, -10 };

    layout reorder_layout(data_types::i8, format::bfyx, {1, 1, 3, 1});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      pooling("pool0", input_info("input0"), pooling_mode::max, {2, 2}, {1, 1}),
                      pooling("pool1", input_info("input1"), pooling_mode::max, {2, 2}, {1, 1}),
                      concatenation("concat",
                                    { input_info("pool0"), input_info("pool1") },
                                    1,
                                    data_types::i8),
                      data("weights", weights),
                      convolution("conv", input_info("concat"), "weights", "", 1, {1, 1}, {1, 1}, {0, 1}, {0, 1}, false),
                      reorder("output", input_info("conv"), reorder_layout) );
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(0);
    int x_size = output_layout.spatial(1);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 1);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, no_exception_in_input_order_opt_b_fs_yx_fsv16_with_conv_port2) {
    auto& engine = get_test_engine();

    auto concat_input0 = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 1, 24, 6, 6 }});
    auto concat_input1 = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 1, 48, 6, 6 }});
    auto concat_input2 = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 1, 96, 6, 6 }});
    auto concat_input3 = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 1, 128, 6, 6 }});
    auto weights0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 296, 296, 1, 1 } });

    std::vector<float> concat_input0_data(concat_input0->get_layout().count());
    std::vector<float> concat_input1_data(concat_input1->get_layout().count());
    std::vector<float> concat_input2_data(concat_input2->get_layout().count());
    std::vector<float> concat_input3_data(concat_input3->get_layout().count());
    std::vector<float> weights0_data(weights0->get_layout().count());

    std::iota(concat_input0_data.begin(), concat_input0_data.end(), 0.f);
    std::iota(concat_input1_data.begin(), concat_input1_data.end(), 0.f);
    std::iota(concat_input2_data.begin(), concat_input2_data.end(), 0.f);
    std::iota(concat_input3_data.begin(), concat_input3_data.end(), 0.f);
    std::iota(weights0_data.begin(), weights0_data.end(), 0.f);

    set_values(concat_input0, concat_input0_data);
    set_values(concat_input1, concat_input1_data);
    set_values(concat_input2, concat_input2_data);
    set_values(concat_input3, concat_input3_data);
    set_values(weights0, weights0_data);

    layout reorder_layout(data_types::f32, format::b_fs_yx_fsv16, {1, 296, 6, 6});

    topology topology(input_layout("concat_input0", concat_input0->get_layout()),
                      input_layout("concat_input1", concat_input1->get_layout()),
                      input_layout("concat_input2", concat_input2->get_layout()),
                      input_layout("concat_input3", concat_input3->get_layout()),
                      concatenation("concat",
                                    { input_info("concat_input0"), input_info("concat_input1"), input_info("concat_input2"), input_info("concat_input3")  },
                                    1,
                                    data_types::f32),
                      pooling("pooling", input_info("concat"), pooling_mode::max, { 2, 2 }, { 1, 1 }),
                      data("weights0", weights0),
                      convolution("conv0", input_info("pooling"), "weights0", "", 1, { 1, 1 }, {1, 1}, {0, 0}, {0, 0}, false),
                      permute("permute", input_info("conv0"), {0, 1, 2, 3}));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("concat_input0", concat_input0);
    network.set_input_data("concat_input1", concat_input1);
    network.set_input_data("concat_input2", concat_input2);
    network.set_input_data("concat_input3", concat_input3);

    ASSERT_NO_FATAL_FAILURE(network.execute());
}

using TestParamType_concat = ::testing::tuple<size_t,   // 0 - Input Batch size
        std::vector<size_t>,                            // 1 - Inputs Features Sizes
        size_t,                                         // 2 - Input Y Size
        size_t,                                         // 3 - Input X Size
        bool>;                                          // 4 - is_caching_test

struct concat_gpu : public ::testing::TestWithParam<TestParamType_concat>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_concat> param_info)
    {
        std::string in;
        for (size_t i = 0; i < testing::get<1>(param_info.param).size() - 1; i++) {
            in += std::to_string(testing::get<1>(param_info.param)[i]) + "_";
        }
        in += std::to_string(testing::get<1>(param_info.param)[testing::get<1>(param_info.param).size() - 1]);

        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + in + "x" + std::to_string(testing::get<2>(param_info.param))
               + 'x' + std::to_string(testing::get<3>(param_info.param))
               + "is_caching_test" + std::to_string(testing::get<4>(param_info.param));
    }
};

using TestParamType_concat_axis3 = ::testing::tuple<size_t,   // 0 - Input Batch size
        size_t,                                               // 1 - Inputs Features Sizes
        size_t,                                               // 2 - Input Y Size
        std::vector<size_t>>;                                 // 3 - Input X Size

struct concat_axis3_gpu : public ::testing::TestWithParam<TestParamType_concat_axis3>
{
    tests::random_generator rg;
    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_concat_axis3> param_info)
    {
        std::string in;
        for (size_t i = 0; i < testing::get<3>(param_info.param).size() - 1; i++) {
            in += std::to_string(testing::get<3>(param_info.param)[i]) + "_";
        }
        in += std::to_string(testing::get<3>(param_info.param)[testing::get<3>(param_info.param).size() - 1]);

        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + in + "x" + std::to_string(testing::get<1>(param_info.param))
               + 'x' + std::to_string(testing::get<2>(param_info.param));
    }
};

static const auto concat_gpu_all_params = ::testing::Values(
    // Input Batch, Input Features, Input Y, Input X
    TestParamType_concat(2, { 2, 15 }, 2, 1, false),
    TestParamType_concat(2, { 2, 31 }, 2, 1, false),
    TestParamType_concat(2, { 2, 32 }, 2, 1, false),
    TestParamType_concat(2, { 2, 37 }, 2, 1, false),
    TestParamType_concat(2, { 2, 63 }, 2, 1, false),
    TestParamType_concat(2, { 2, 64 }, 2, 1, false),
    TestParamType_concat(2, { 2, 65 }, 2, 1, false),
    TestParamType_concat(2, { 2, 75 }, 2, 1, false),
    TestParamType_concat(2, { 15, 2 }, 2, 1, false),
    TestParamType_concat(2, { 31, 2 }, 2, 1, false),
    TestParamType_concat(2, { 32, 2 }, 2, 1, false),
    TestParamType_concat(2, { 37, 2 }, 2, 1, false),
    TestParamType_concat(2, { 63, 2 }, 2, 1, false),
    TestParamType_concat(2, { 64, 2 }, 2, 1, false),
    TestParamType_concat(2, { 65, 2 }, 2, 1, false),
    TestParamType_concat(2, { 75, 2 }, 2, 1, false),
    TestParamType_concat(2, { 2, 15 }, 1, 2, false),
    TestParamType_concat(2, { 2, 31 }, 1, 2, false),
    TestParamType_concat(2, { 2, 32 }, 1, 2, false),
    TestParamType_concat(2, { 2, 37 }, 1, 2, false),
    TestParamType_concat(2, { 2, 63 }, 1, 2, false),
    TestParamType_concat(2, { 2, 64 }, 1, 2, false),
    TestParamType_concat(2, { 2, 65 }, 1, 2, false),
    TestParamType_concat(2, { 2, 75 }, 1, 2, false),
    TestParamType_concat(2, { 15, 2 }, 1, 2, false),
    TestParamType_concat(2, { 31, 2 }, 1, 2, false),
    TestParamType_concat(2, { 32, 2 }, 1, 2, false),
    TestParamType_concat(2, { 37, 2 }, 1, 2, false),
    TestParamType_concat(2, { 63, 2 }, 1, 2, false),
    TestParamType_concat(2, { 64, 2 }, 1, 2, false),
    TestParamType_concat(2, { 65, 2 }, 1, 2, false),
    TestParamType_concat(2, { 75, 2 }, 1, 2, false),
    TestParamType_concat(2, { 32, 32 }, 1, 1, false),
    TestParamType_concat(2, { 64, 64 }, 1, 1, false),
    TestParamType_concat(2, { 2, 2, 2 }, 1, 1, false),
    TestParamType_concat(2, { 2, 32, 2 }, 1, 1, false),
    TestParamType_concat(2, { 31, 32, 32 }, 1, 1, false),
    TestParamType_concat(2, { 32, 31, 2 }, 1, 1, false),
    TestParamType_concat(2, { 32, 31, 32 }, 1, 1, false),
    TestParamType_concat(2, { 32, 32, 32 }, 1, 1, false),
    TestParamType_concat(2, { 33, 32, 32 }, 1, 1, false),
    TestParamType_concat(2, { 33, 3, 3 }, 1, 1, false),
    TestParamType_concat(2, { 33, 3, 33 }, 1, 1, false),
    TestParamType_concat(2, { 64, 64, 64, 64 }, 1, 1, false)
);

template <typename Type>
struct concat_gpu_4d : public concat_gpu {
public:

    void test(format::type fmt) {
        auto data_type = ov::element::from<Type>();

        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory::ptr> in_memory;
        std::vector<input_info> input_ids;
        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = rg.generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back(input_info("input" + std::to_string(i)));
        }

        topology.add(concatenation("concat", input_ids, 1));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        network network(engine, topology, config);

        for (size_t i = 0; i < in_features.size(); i++) {
            network.set_input_data(input_ids[i].pid, in_memory[i]);
        }

        auto outputs = network.execute();

        auto out_mem = outputs.at("concat").get_memory();
        cldnn::mem_lock<Type> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num; bi++) {
            size_t f_sum = 0;
            for (size_t in_i = 0; in_i < in_features.size(); in_i++) {
                for (size_t fi = 0; fi < in_features[in_i]; fi++) {
                    for (size_t yi = 0; yi < input_y; yi++) {
                        for (size_t xi = 0; xi < input_x; xi++) {
                            auto output_coords = tensor(batch(bi), feature(f_sum + fi), spatial(xi, yi, 0, 0));
                            auto output_offset = out_mem->get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = out_ptr[output_offset];
                            ASSERT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << f_sum + fi << "(input " << in_i << "), y=" << yi << ", x=" << xi;
                        }
                    }
                }
                f_sum += in_features[in_i];
            }
        }
    }
};

// Test case for axis=3 case in 4D
template <typename Type>
struct concat_gpu_4d_axis3 : public concat_axis3_gpu {
public:

    void test(format::type fmt) {
        auto data_type = ov::element::from<Type>();

        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const size_t in_feature = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const std::vector<size_t> input_x = testing::get<3>(GetParam());
        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory::ptr> in_memory;
        std::vector<input_info> input_ids;
        for (size_t i = 0; i < input_x.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_feature),
                               static_cast<int32_t>(input_x[i]),
                               static_cast<int32_t>(input_y));
            auto data = rg.generate_random_4d<Type>(batch_num, in_feature, input_y, input_x[i], -1, 1);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_feature; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x[i]; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back(input_info("input" + std::to_string(i)));
        }

        topology.add(concatenation("concat", input_ids, 3));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        network network(engine, topology, config);

        for (size_t i = 0; i < input_x.size(); i++) {
            network.set_input_data(input_ids[i].pid, in_memory[i]);
        }

        auto outputs = network.execute();

        auto out_mem = outputs.at("concat").get_memory();
        cldnn::mem_lock<Type> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num; bi++) {
            for (size_t fi = 0; fi < in_feature; fi++) {
                for (size_t yi = 0; yi < input_y; yi++) {
                    size_t x_sum = 0;
                    for (size_t in_i = 0; in_i < input_x.size(); in_i++) {
                        for (size_t xi = 0; xi < input_x[in_i]; xi++) {
                            auto output_coords = tensor(batch(bi), feature(fi), spatial((x_sum + xi), yi, 0, 0));
                            auto output_offset = out_mem->get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = out_ptr[output_offset];
                            ASSERT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << fi << ", y=" << yi << ", x=" << x_sum + xi << "(input " << in_i << ")";
                        }
                        x_sum += input_x[in_i];
                    }
                }
            }
        }
    }
};


using concat_gpu_4d_f16 = concat_gpu_4d<ov::float16>;
using concat_gpu_4d_i8 = concat_gpu_4d<int8_t>;
using concat_gpu_4d_u8 = concat_gpu_4d<uint8_t>;

TEST_P(concat_gpu_4d_f16, fs_b_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::fs_b_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_gpu_4d_f16,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_i8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

TEST_P(concat_gpu_4d_i8, b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_gpu_4d_i8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_u8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_gpu_4d_u8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

using concat_gpu_4d_axis3_f16 = concat_gpu_4d_axis3<ov::float16>;

TEST_P(concat_gpu_4d_axis3_f16, fs_b_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::fs_b_yx_fsv32));
}

TEST_P(concat_gpu_4d_axis3_f16, b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

TEST_P(concat_gpu_4d_axis3_f16, bs_fs_yx_bsv16_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::bs_fs_yx_bsv16_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_gpu_4d_axis3_f16,
                        ::testing::Values(
                            TestParamType_concat_axis3(2, 16, 2, { 2, 3 }),
                            TestParamType_concat_axis3(2, 19, 2, { 2, 3, 2 }),
                            TestParamType_concat_axis3(2, 32, 2, { 2, 3, 2, 1 }),
                            TestParamType_concat_axis3(2, 35, 2, { 3, 2, 3, 2 })
                        ),
                        concat_axis3_gpu::PrintToStringParamName);

template <typename Type, typename OutputT>
struct concat_id_conv_gpu_4d : public concat_gpu {
public:

    void test(format::type fmt) {
        auto data_type = ov::element::from<Type>();

        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory::ptr> in_memory;
        std::vector<input_info> input_ids;
        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = rg.generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -128, 128);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back(input_info("input" + std::to_string(i)));
        }

        topology.add(concatenation("concat", input_ids, 1));
        // Add identity convolution
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        weights_mem->fill(get_test_stream());
        get_test_stream().finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, get_test_stream());
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights", weights_mem));
        topology.add(convolution("conv", input_info("concat"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto conv_forcing = ov::intel_gpu::ImplementationDesc{ fmt, std::string() };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {primitive_id("conv"), conv_forcing} }));
        network network(engine, topology, config);

        for (size_t i = 0; i < in_features.size(); i++) {
            network.set_input_data(input_ids[i].pid, in_memory[i]);
        }

        auto outputs = network.execute();

        auto out_mem = outputs.at("conv").get_memory();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());
        ASSERT_EQ(out_mem->get_layout().format, fmt);

        for (size_t bi = 0; bi < batch_num; bi++) {
            size_t f_sum = 0;
            for (size_t in_i = 0; in_i < in_features.size(); in_i++) {
                for (size_t fi = 0; fi < in_features[in_i]; fi++) {
                    for (size_t yi = 0; yi < input_y; yi++) {
                        for (size_t xi = 0; xi < input_x; xi++) {
                            auto output_coords = tensor(batch(bi), feature(f_sum + fi), spatial(xi, yi, 0, 0));
                            auto output_offset = out_mem->get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = static_cast<Type>(out_ptr[output_offset]);
                            ASSERT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << f_sum + fi << "(input " << in_i << "), y=" << yi << ", x=" << xi;
                        }
                    }
                }
                f_sum += in_features[in_i];
            }
        }
    }
};

using concat_id_conv_gpu_4d_f16 = concat_id_conv_gpu_4d<ov::float16, ov::float16>;
using concat_id_conv_gpu_4d_i8 = concat_id_conv_gpu_4d<int8_t, float>;

TEST_P(concat_id_conv_gpu_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_id_conv_gpu_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(2, { 2, 32 }, 2, 1, false),
                            TestParamType_concat(2, { 31, 64 }, 2, 2, false),
                            TestParamType_concat(2, { 15, 15, 16 }, 2, 1, false),
                            TestParamType_concat(2, { 16, 15, 16 }, 2, 2, false),
                            TestParamType_concat(2, { 15, 2, 16, 64 }, 1, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_id_conv_gpu_4d_i8, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_id_conv_gpu_4d_i8,
                        ::testing::Values(
                            TestParamType_concat(2, { 2, 32 }, 2, 1, false),
                            TestParamType_concat(2, { 31, 64 }, 2, 2, false),
                            TestParamType_concat(2, { 15, 15, 16 }, 2, 1, false),
                            TestParamType_concat(2, { 16, 15, 16 }, 2, 2, false),
                            TestParamType_concat(2, { 15, 2, 16, 64 }, 1, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);

template <typename Type>
struct concat_gpu_4d_implicit : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, ExecutionConfig config) {
        auto data_type = ov::element::from<Type>();
        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        const bool is_caching_test = testing::get<4>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        std::vector<input_info> pooling_ids;

        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            topology.add(pooling("pool" +  std::to_string(i), input_info("input" + std::to_string(i)), pooling_mode::max, {1, 1}, {1, 1}));

            input_ids.push_back("input" + std::to_string(i));
            pooling_ids.push_back(input_info("pool" + std::to_string(i)));
        }

        topology.add(concatenation("concat", pooling_ids, 1));
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        weights_mem->fill(get_test_stream());
        get_test_stream().finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, get_test_stream());
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", input_info("concat"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(pooling("pool_final", input_info("conv"), pooling_mode::max, {1, 1}, {1, 1}));
        topology.add(reorder("reorder", input_info("pool_final"), layout(data_type, format::bfyx, {(int32_t)batch_num, (int32_t)output_f, (int32_t)input_y, (int32_t)input_x})));

        cldnn::network::ptr concat_network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        for (size_t i = 0; i < in_features.size(); i++) {
            concat_network->set_input_data(input_ids[i], in_memory[i]);
        }
        auto outputs = concat_network->execute();

        bool concat_opt_enabled = config.get_optimize_data();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network->get_primitive("concat"))->can_be_optimized();
        EXPECT_EQ(concat_opt_enabled, concat_opt_result);

        return outputs.at("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input(in_features.size());
        for (size_t i = 0; i < in_features.size(); ++i) {
            input[i] = rg.generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
        }
        return input;
    }

    void test(format::type fmt) {
        auto input = generate_input();

        // implicit concat
        ExecutionConfig config1 = get_test_default_config(get_test_engine());
        config1.set_property(ov::intel_gpu::optimize_data(true));
        auto out_mem1 = run_concat_network(input, fmt, config1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, get_test_stream());

        // explicit concat
        ExecutionConfig config2 = get_test_default_config(get_test_engine());
        config2.set_property(ov::intel_gpu::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, config2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, get_test_stream());

        ASSERT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        ASSERT_EQ(diff_count, 0);
    }
};

using concat_implicit_gpu_4d_f16 = concat_gpu_4d_implicit<ov::float16>;
using concat_implicit_gpu_4d_i8 = concat_gpu_4d_implicit<int8_t>;

TEST_P(concat_implicit_gpu_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 16, 16 }, 2, 2, false),
                            TestParamType_concat(1, { 16, 8 }, 2, 2, false),
                            TestParamType_concat(1, { 8, 16 }, 2, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(export_import,
                        concat_implicit_gpu_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 8, 16 }, 2, 2, true)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_implicit_gpu_4d_i8, input_order_opt_b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(concat_gpu_onednn, basic_input_types) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input4 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });

    set_values<float>(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values<float>(input1, { 11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f });
    set_values<float>(input2, { 21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f });
    set_values<float>(input3, { 31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f });
    set_values<float>(input4, { 41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            input_layout("input4", input4->get_layout()),
            concatenation("concat",
                          { input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3"), input_info("input4") },
                          1,
                          data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc impl = { format::bfyx, std::string(""), impl_types::onednn };

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "concat" }));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"concat", impl} }));
    network network(engine, topology, cfg);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 5);
    ASSERT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        ASSERT_EQ(output_vec[x], output_ptr[x]);
    }
}

template <typename Type>
struct concat_gpu_4d_implicit_onednn : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, ExecutionConfig config) {
        auto data_type = ov::element::from<Type>();
        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        std::vector<input_info> pooling_ids;

        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            topology.add(pooling("pool" +  std::to_string(i), input_info("input" + std::to_string(i)), pooling_mode::max, {1, 1}, {1, 1}));

            input_ids.push_back("input" + std::to_string(i));
            pooling_ids.push_back(input_info("pool" + std::to_string(i)));
        }

        topology.add(concatenation("concat", pooling_ids, 1));
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        auto& stream = get_test_stream();
        weights_mem->fill(stream);
        stream.finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, stream);
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", input_info("concat"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(pooling("pool_final", input_info("conv"), pooling_mode::max, {1, 1}, {1, 1}));
        topology.add(reorder("reorder", input_info("pool_final"), layout(data_type, format::bfyx, {(int32_t)batch_num, (int32_t)output_f, (int32_t)input_y, (int32_t)input_x})));

        network concat_network(engine, topology, config);
        for (size_t i = 0; i < in_features.size(); i++) {
            concat_network.set_input_data(input_ids[i], in_memory[i]);
        }
        auto outputs = concat_network.execute();

        bool concat_opt_enabled = config.get_optimize_data();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network.get_primitive("concat"))->node->can_be_optimized();
        EXPECT_EQ(concat_opt_enabled, concat_opt_result);

        return outputs.at("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input(in_features.size());
        for (size_t i = 0; i < in_features.size(); ++i) {
            input[i] = rg.generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
        }
        return input;
    }

    void test(format::type fmt) {
        auto& engine = get_test_engine();
        auto& stream = get_test_stream();
        if (!engine.get_device_info().supports_immad) {
            // This case is only for device that uses onednn.
            return;
        }
        auto input = generate_input();

        // implicit concat
        ExecutionConfig config1 = get_test_default_config(engine);
        config1.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc impl = { fmt, std::string(""), impl_types::onednn };
        config1.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", impl} }));

        auto out_mem1 = run_concat_network(input, fmt, config1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, stream);

        // explicit concat
        ExecutionConfig config2 = get_test_default_config(engine);
        config2.set_property(ov::intel_gpu::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, config2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, stream);

        ASSERT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        ASSERT_EQ(diff_count, 0);
    }
};


using concat_implicit_gpu_onednn_4d_f16 = concat_gpu_4d_implicit_onednn<ov::float16>;
using concat_implicit_gpu_onednn_4d_i8 = concat_gpu_4d_implicit_onednn<int8_t>;

TEST_P(concat_implicit_gpu_onednn_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_onednn_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 16, 16 }, 2, 2, false),
                            TestParamType_concat(1, { 16, 8 }, 2, 2, false),
                            TestParamType_concat(1, { 8, 16 }, 2, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_implicit_gpu_onednn_4d_i8, input_order_opt_b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_onednn_4d_i8,
                        ::testing::Values(
                            TestParamType_concat(1, { 32, 32 }, 2, 2, false),
                            TestParamType_concat(1, { 32, 8 }, 2, 2, false),
                            TestParamType_concat(1, { 8, 32 }, 2, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);


template <typename Type>
struct concat_gpu_4d_implicit_mix_types_onednn : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, ExecutionConfig config) {
        auto data_type = ov::element::from<Type>();
        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        std::vector<input_info> pooling_ids;

        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            topology.add(pooling("pool" +  std::to_string(i), input_info("input" + std::to_string(i)), pooling_mode::max, {1, 1}, {1, 1}));

            input_ids.push_back("input" + std::to_string(i));
            pooling_ids.push_back(input_info("pool" + std::to_string(i)));
        }

        topology.add(concatenation("concat", pooling_ids, 1));
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        auto& stream = get_test_stream();
        weights_mem->fill(stream);
        stream.finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, stream);
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }

        std::vector<input_info> concat_ids;

        concat_ids.push_back(input_info("input1"));
        concat_ids.push_back(input_info("pool_final"));

        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", input_info("concat"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(pooling("pool_final", input_info("conv"), pooling_mode::max, {1, 1}, {1, 1}));
        topology.add(concatenation("concat_final", concat_ids, 1));
        topology.add(reorder("reorder", input_info("concat_final"), layout(data_type, format::byxf, {(int32_t)batch_num, (int32_t)output_f, (int32_t)input_y, (int32_t)input_x})));

        network concat_network(engine, topology, config);
        for (size_t i = 0; i < in_features.size(); i++) {
            concat_network.set_input_data(input_ids[i], in_memory[i]);
        }
        auto outputs = concat_network.execute();

        bool concat_opt_enabled = config.get_optimize_data();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network.get_primitive("concat_final"))->node->can_be_optimized();
        EXPECT_EQ(concat_opt_enabled, concat_opt_result);

        return outputs.at("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input(in_features.size());
        for (size_t i = 0; i < in_features.size(); ++i) {
            input[i] = rg.generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -10, 10);
        }
        return input;
    }

    void test(format::type fmt) {
        auto& engine = get_test_engine();
        auto& stream = get_test_stream();
        if (!engine.get_device_info().supports_immad) {
            // This case is only for device that uses onednn.
            return;
        }
        auto input = generate_input();

        // implicit concat
        ExecutionConfig config1 = get_test_default_config(engine);
        config1.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc impl = { format::bfyx, std::string(""), impl_types::onednn };
        config1.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", impl} }));

        auto out_mem1 = run_concat_network(input, fmt, config1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, stream);

        // explicit concat
        ExecutionConfig config2 = get_test_default_config(engine);
        config2.set_property(ov::intel_gpu::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, config2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, stream);

        ASSERT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        ASSERT_EQ(diff_count, 0);
    }
};

using concat_implicit_gpu_onednn_4d_mix_i8 = concat_gpu_4d_implicit_mix_types_onednn<ov::float16>;

TEST_P(concat_implicit_gpu_onednn_4d_mix_i8, input_order_opt_b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_onednn_4d_mix_i8,
                        ::testing::Values(
                          TestParamType_concat(1, { 8, 32 }, 2, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);


template <typename Type>
struct concat_gpu_4d_explicit : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, ExecutionConfig config) {
        auto data_type = ov::element::from<Type>();
        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam()); // only use first element.
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = in_features[0];

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;

        // input0 --- eltwise1 --- concat --- reorder
        //          /            /
        // input1 --            /
        //                     /
        // input2 --- eltwise2 -------------- conv
        //          /
        // input3 --
        for (size_t i = 0; i < 4; i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(output_f),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, format::bfyx, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < output_f; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            input_ids.push_back("input" + std::to_string(i));
        }

        topology.add(eltwise("eltwise1", {input_info(input_ids[0]), input_info(input_ids[1])}, eltwise_mode::sum));
        topology.add(eltwise("eltwise2", {input_info(input_ids[2]), input_info(input_ids[3])}, eltwise_mode::sum));

        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        auto& stream = get_test_stream();
        weights_mem->fill(stream);
        stream.finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, stream);
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", input_info("eltwise2"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(concatenation("concat", {input_info("eltwise1"), input_info("eltwise2")}, 1));
        topology.add(reorder("reorder", input_info("concat"), layout(data_types::f32, format::bfyx, {(int32_t)batch_num, (int32_t)(output_f * 2), (int32_t)input_y, (int32_t)input_x})));

        network concat_network(engine, topology, config);
        for (size_t i = 0; i < 4; i++) {
            concat_network.set_input_data(input_ids[i], in_memory[i]);
        }
        auto outputs = concat_network.execute();

        bool concat_opt_enabled = config.get_optimize_data();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network.get_primitive("concat"))->node->can_be_optimized();

        // If sibling is using onednn impl and batch > 1, the onednn impl cannot process the implicit concat'ed buffer.
        // Onednn impls can process implicit concat'ed buffer only through buffer pointer manipulation.
        if (concat_opt_enabled && batch_num > 1) concat_opt_result = !concat_opt_result;
        EXPECT_EQ(concat_opt_enabled, concat_opt_result);

        return outputs.at("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> inputs(4);
        for (size_t i = 0; i < 4; ++i) {
            inputs[i] = rg.generate_random_4d<Type>(batch_num, in_features[0], input_y, input_x, -1, 1);
        }
        return inputs;
    }

    void test(format::type fmt) {
        auto& engine = get_test_engine();
        auto& stream = get_test_stream();
        if (!engine.get_device_info().supports_immad) {
            // This case is only for device that uses onednn.
            return;
        }
        auto input = generate_input();

        // implicit concat when batch size is 1.
        ExecutionConfig config1 = get_test_default_config(engine);
        config1.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc impl = { fmt, std::string(""), impl_types::onednn };
        config1.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", impl}}));

        auto out_mem1 = run_concat_network(input, fmt, config1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, stream);

        // explicit concat
        ExecutionConfig config2 = get_test_default_config(engine);
        config2.set_property(ov::intel_gpu::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, config2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, stream);

        ASSERT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        ASSERT_EQ(diff_count, 0);
    }
};


using concat_no_implicit_gpu_onednn_4d_f16 = concat_gpu_4d_explicit<ov::float16>;

TEST_P(concat_no_implicit_gpu_onednn_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_no_implicit_gpu_onednn_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 16 }, 2, 2, false),
                            TestParamType_concat(2, { 16 }, 2, 2, false)
                        ),
                        concat_gpu::PrintToStringParamName);
#endif
