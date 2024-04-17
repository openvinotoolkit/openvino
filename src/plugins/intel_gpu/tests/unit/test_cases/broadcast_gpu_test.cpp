// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_set.hpp"
#include "test_utils.h"
#include "openvino/reference/tile.hpp"
#include "openvino/reference/broadcast.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/broadcast.hpp>

#include "broadcast_inst.h"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

static ov::AxisSet get_broadcast_axes(const ov::AxisSet& axes_mapping_val, const size_t target_shape_size) {
    ov::AxisSet broadcast_axes;

    std::vector<size_t> axes(target_shape_size);
    std::iota(axes.begin(), axes.end(), 0);
    for (auto i = axes_mapping_val.rbegin(); i != axes_mapping_val.rend(); ++i) {
        axes.erase(axes.begin() + *i);
    }
    broadcast_axes.insert(axes.begin(), axes.end());

    return broadcast_axes;
}


template<typename T>
void start_broadcast_test(format cldnn_format, data_types cldnn_data_type, ov::Shape output_shape,
                          ov::Shape input_shape, ov::AxisSet axes_mapping) {
    size_t input_data_size = ov::shape_size(input_shape);
    ASSERT_GE(input_data_size, (size_t)1);
    std::vector<T> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((T)i);
    }

    size_t output_data_size = accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(output_data_size, (size_t)1);
    std::vector<T> output_data(output_data_size);
    ov::reference::broadcast(reinterpret_cast<const char*>(input_data.data()),
                             reinterpret_cast<char*>(output_data.data()),
                             input_shape,
                             output_shape,
                             get_broadcast_axes(axes_mapping, output_shape.size()),
                             sizeof(T));

    ASSERT_EQ(output_data.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({input_shape, cldnn_data_type, format::bfyx});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), cldnn_format, cldnn_data_type));
    topology.add(broadcast("broadcast", input_info("reorder"), output_shape, axes_mapping));
    topology.add(reorder("output", input_info("broadcast"), format::bfyx, cldnn_data_type));

    set_values(input, input_data);

    ExecutionConfig cfg = get_test_default_config(engine);
    network network(engine, topology, cfg);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ov::shape_size(output_shape); ++i) {
        ASSERT_EQ(output_ptr[i], output_data[i]) << " i = " << i;
    }
}
template<typename inT, typename outT>
void start_broadcast_test_dynamic(format input_format,
                                  data_types input_data_type,
                                  data_types output_data_type,
                                  ov::Shape output_shape,
                                  ov::Shape input_data_shape,
                                  ov::AxisSet axes_mapping,
                                  bool is_output_static = false,
                                  impl_types impl_type = impl_types::any,
                                  bool optimize = false) {
    size_t input_data_size = accumulate(input_data_shape.rbegin(), input_data_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(input_data_size, (size_t)1);
    std::vector<inT> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((inT)i);
    }

    size_t output_data_size = accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(output_data_size, (size_t)1);
    std::vector<inT>  output_data_tmp(output_data_size);
    std::vector<outT> output_data;
    ov::reference::broadcast(reinterpret_cast<const char*>(input_data.data()),
                             reinterpret_cast<char*>(output_data_tmp.data()),
                             ov::Shape(input_data_shape.begin(), input_data_shape.end()),
                             ov::Shape(output_shape.begin(), output_shape.end()),
                             get_broadcast_axes(axes_mapping, output_shape.size()),
                             sizeof(inT));

    ASSERT_EQ(output_data_tmp.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));
    for (auto i : output_data_tmp) {
        output_data.push_back((outT)i);
    }

    int64_t input_rank = input_data_shape.size();
    ASSERT_EQ(input_rank, axes_mapping.size());
    auto fmt = format::get_default_format(input_rank);

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ov::PartialShape(input_data_shape), input_data_type, fmt});

    topology topology;
    memory::ptr target_shape_mem = nullptr;
    if (is_output_static) {
        auto in_layout = layout(ov::PartialShape::dynamic(input_rank), input_data_type, fmt);
        topology.add(input_layout("input", in_layout));
        topology.add(reorder("reorder", input_info("input"), input_format, input_data_type));
        topology.add(broadcast("broadcast", input_info("reorder"), output_shape, axes_mapping));
        topology.add(reorder("output",
                             input_info("broadcast"),
                             fmt,
                             output_data_type,
                             std::vector<float>(),
                             cldnn::reorder_mean_mode::subtract,
                             cldnn::padding(),
                             true));
    } else {
        auto in_layout = layout(ov::PartialShape::dynamic(input_rank), input_data_type, fmt);
        auto target_shape_layout = layout(ov::PartialShape{input_rank}, data_types::i32, fmt);
        target_shape_mem = engine.allocate_memory(target_shape_layout);
        topology.add(input_layout("input", in_layout));
        topology.add(input_layout("target_shape", target_shape_layout));
        topology.add(reorder("reorder", input_info("input"), input_format, input_data_type));
        topology.add(
            broadcast("broadcast", input_info("reorder"), input_info("target_shape"), axes_mapping));
        topology.add(reorder("output",
                             input_info("broadcast"),
                             fmt,
                             output_data_type,
                             std::vector<float>(),
                             cldnn::reorder_mean_mode::subtract,
                             cldnn::padding(),
                             true));
        std::vector<int32_t> target_shape_data;
        for (auto out_shape : output_shape) {
            target_shape_data.push_back(static_cast<int32_t>(out_shape));
        }
        set_values<int32_t>(target_shape_mem, target_shape_data);
    }

    ExecutionConfig config = get_test_default_config(engine);
    if (optimize) {
        config.set_property(ov::intel_gpu::optimize_data(true));
    }

    const bool force_impl = impl_type != impl_types::any;
    if (force_impl) {
        auto forcing_map = ov::intel_gpu::ImplForcingMap{{"broadcast", {input_format, "", impl_type}},
                                                         {"reorder", {fmt, "", impl_type}}};
        config.set_property(ov::intel_gpu::force_implementations(forcing_map));
    }

    set_values(input, input_data);

    network network(engine, topology, config);
    network.set_input_data("input", input);
    if (!is_output_static) {
        network.set_input_data("target_shape", target_shape_mem);
    }

    // In case of impl forcing optimize_data property will set to true and additional
    // reorders optimization pass will be tiggered, so change expected primitive id
    const auto prim_id = (force_impl || optimize) ? "output" : "broadcast";
    auto inst = network.get_primitive(prim_id);
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<outT> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_data_size; ++i) {
        ASSERT_EQ(output_ptr[i], output_data[i]);
    }
}

template<typename T>
void start_broadcast_test_5d(format cldnn_format, data_types cldnn_data_type, ov::Shape output_shape,
                             ov::Shape input_shape, ov::AxisSet axes_mapping, bool is_caching_test=false)
{
    size_t input_data_size = ov::shape_size(input_shape);
    ASSERT_GE(input_data_size, (size_t)1);
    std::vector<T> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((T)i);
    }

    size_t output_data_size = accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(output_data_size, (size_t)1);
    std::vector<T> output_data(output_data_size);
    ov::reference::broadcast(reinterpret_cast<const char*>(input_data.data()),
                             reinterpret_cast<char*>(output_data.data()),
                             input_shape,
                             output_shape,
                             get_broadcast_axes(axes_mapping, output_shape.size()),
                             sizeof(T));

    ASSERT_EQ(output_data.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ input_shape, cldnn_data_type, format::bfzyx });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), cldnn_format, cldnn_data_type));
    topology.add(broadcast("broadcast", input_info("reorder"), output_shape, axes_mapping));
    topology.add(reorder("output", input_info("broadcast"), format::bfzyx, cldnn_data_type));

    set_values(input, input_data);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ov::shape_size(output_shape); ++i) {
        ASSERT_EQ(output_ptr[i], output_data[i]);
    }
}

/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0} */
TEST(broadcast_gpu_float, bfyx_1_to_5_w_b_axes_0) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {5}, {1}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_5_w_b_axes_0) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {5}, {1}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_5_w_b_axes_0) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {5}, {1}, {0});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; */
TEST(broadcast_gpu_float, bfyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {4, 5}, {1}, {0});
}

// dynamic kernel
TEST(broadcast_gpu_float, bfyx_1_to_4x5_w_b_axes_0x1_dynamic) {
    start_broadcast_test_dynamic<float, ov::float16>(format::bfyx, data_types::f32, data_types::f16, {4, 5}, {1, 1}, {0, 1}, false, impl_types::any, true);
}

TEST(broadcast_gpu_float, bfyx_1_to_4x5_w_b_axes_0x1_dynamic_with_static_output) {
    start_broadcast_test_dynamic<float, ov::float16>(format::bfyx, data_types::f32, data_types::f16, {4, 5}, {1, 1}, {0, 1}, true);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_4x5_w_b_axes_0x1_dynamic) {
    start_broadcast_test_dynamic<uint8_t, int32_t>(format::bfyx, data_types::u8, data_types::i32, {4, 5}, {1, 1}, {0, 1});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_4x5_w_b_axes_0x1x2_dynamic_with_static_output) {
    start_broadcast_test_dynamic<uint8_t, int32_t>(format::bfyx, data_types::u8, data_types::i32, {4, 5, 2}, {1, 1, 1}, {0, 1, 2}, true);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_4x5_w_b_axes_0x1_dynamic) {
    start_broadcast_test_dynamic<int64_t, int32_t>(format::bfyx, data_types::i64, data_types::i32, {4, 5}, {1, 1}, {0, 1}, false, impl_types::any, true);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_4x5_w_b_axes_0x1x2x3_dynamic_with_static_output) {
    start_broadcast_test_dynamic<int64_t, int32_t>(format::bfyx, data_types::i64, data_types::i32, {4, 5, 2, 3}, {1, 1, 1, 1}, {0, 1, 2, 3});
}

// dynamic kernel cpu
TEST(broadcast_cpu_impl_float, bfyx_1_to_4x5_w_b_axes_0x1_dynamic) {
    start_broadcast_test_dynamic<float, int32_t>(format::bfyx, data_types::f32, data_types::i32, {4, 5}, {1, 1}, {0, 1}, false, impl_types::cpu);
}

TEST(broadcast_cpu_impl_float, bfyx_1_to_4x5_w_b_axes_0x1_dynamic_with_static_output) {
    start_broadcast_test_dynamic<float, float>(format::bfyx, data_types::f32, data_types::f32, {4, 5}, {1, 1}, {0, 1}, true, impl_types::cpu);
}

TEST(broadcast_cpu_impl_int64_t, bfyx_1_to_4x5_w_b_axes_0x1_dynamic) {
    start_broadcast_test_dynamic<int64_t, int64_t>(format::bfyx, data_types::i64, data_types::i64, {4, 5}, {1, 1}, {0, 1}, false, impl_types::cpu);
}

TEST(broadcast_cpu_impl_int64_t, bfyx_1_to_4x5_w_b_axes_0x1x2x3_dynamic_with_static_output) {
    start_broadcast_test_dynamic<int64_t, int32_t>(format::bfyx, data_types::i64, data_types::i32, {4, 5, 2, 3}, {1, 1, 1, 1}, {0, 1, 2, 3}, false, impl_types::cpu);
}

/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; */
TEST(broadcast_gpu_float, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {3, 4, 5}, {1}, {1});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {3, 4, 5}, {1}, {1});
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {3, 4, 5}, {1}, {1});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; */
TEST(broadcast_gpu_float, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {1}, {2});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0} */
TEST(broadcast_gpu_float, bfyx_1_to_5_w_o_b_axes) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {5}, {1}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_5_w_o_b_axes) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {5}, {1}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_5_w_o_b_axes) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {5}, {1}, {0});
}



/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; */
TEST(broadcast_gpu_float, bfyx_1x1_to_4x5_w_o_b_axes) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {4, 5}, {1, 1}, {0, 1});
}

TEST(broadcast_gpu_uint8_t, bfyx_1x1_to_4x5_w_o_b_axes) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {4, 5}, {1, 1}, {0, 1});
}

TEST(broadcast_gpu_int64_t, bfyx_1x1_to_4x5_w_o_b_axes) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {4, 5}, {1, 1}, {0, 1});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0} */
TEST(broadcast_gpu_float, bfyx_3_to_2x3_w_b_axes_0) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3}, {3}, {1});
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3_w_b_axes_0) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3}, {3}, {1});
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3_w_b_axes_0) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3}, {3}, {1});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0} */
TEST(broadcast_gpu_float, bfyx_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3}, {2}, {0});
}



/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; */
TEST(broadcast_gpu_float, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_int64_t, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {3, 4}, {1, 2});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                           5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0}; */
TEST(broadcast_gpu_float, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int64_t, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2, 4}, {0, 2});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                           4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0}; */
TEST(broadcast_gpu_float, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int64_t, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2, 3}, {0, 1});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                           1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0} */
TEST(broadcast_gpu_float, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_uint8_t, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_int64_t, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {4}, {2});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                           1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0} */
TEST(broadcast_gpu_float, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {3}, {1});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0} */
TEST(broadcast_gpu_float, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2}, {0});
}


/* Expected golden_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                           25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                           37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                           49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                           1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                           25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                           37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                           49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0} */
TEST(broadcast_gpu_float, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 4, 5}, {1, 2, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 4, 5}, {1, 2, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 4, 5}, {1, 2, 3});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                           31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                           31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                           31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0} */
TEST(broadcast_gpu_float, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                           16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                           16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                           26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                           26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0} */
TEST(broadcast_gpu_float, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                           9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                           13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                           15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                           17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                           19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                           21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                           23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0} */
TEST(broadcast_gpu_float, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0} */
TEST(broadcast_gpu_float, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {4, 5}, {2, 3});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0} */
TEST(broadcast_gpu_float, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 5}, {1, 3});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                           9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                           9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0 */
TEST(broadcast_gpu_float, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_int64_t, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 4}, {1, 2});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0}; */
TEST(broadcast_gpu_float, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_int64_t, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 5}, {0, 3});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0} */
TEST(broadcast_gpu_float, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int64_t, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 4}, {0, 2});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                           4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                           5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                           6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                           6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0}; */
TEST(broadcast_gpu_float, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int64_t, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3}, {0, 1});
}


/* Expected golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0} */
TEST(broadcast_gpu_float, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_uint8_t, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_int64_t, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {5}, {3});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0} */
TEST(broadcast_gpu_float, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_uint8_t, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_int64_t, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {4}, {2});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                           3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0} */
TEST(broadcast_gpu_float, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3}, {1});
}


/* Expected golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0} */
TEST(broadcast_gpu_float, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2}, {0});
}

/* Expected golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0 } */
TEST(broadcast_gpu_float, bfzyx_1_to_5_w_b_axes_0) {
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 5 }, { 1 }, { 0 });
}

TEST(broadcast_gpu_uint8_t, bfzyx_1_to_5_w_b_axes_0) {
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 5 }, { 1 }, { 0 });
}

TEST(broadcast_gpu_int64_t, bfzyx_1_to_5_w_b_axes_0) {
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 5 }, { 1 }, { 0 });
}


/* Expected golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } */
TEST(broadcast_gpu_float, bfzyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 4, 5 }, { 1 }, { 1 });
}

TEST(broadcast_gpu_uint8_t, bfzyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 4, 5 }, { 1 }, { 1 });
}

TEST(broadcast_gpu_int64_t, bfzyx_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 4, 5 }, { 1 }, { 1 });
}


/* Expected golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } */
TEST(broadcast_gpu_float, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 2, 3, 4, 5, 2 }, { 1 }, { 2 });
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 2, 3, 4, 5, 2 }, { 1 }, { 2 });
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 2, 3, 4, 5, 2 }, { 1 }, { 2 });
}


/* BLOCKED FORMAT TEST CASES */
TEST(broadcast_gpu_float, b_fs_yx_fsv16_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1x38x1x1_to_1x38x1x5_w_b_axes_0) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {1, 38, 1, 5}, {1, 38, 1, 1}, {0, 1, 2, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {4, 5}, {1}, {0});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {4, 5}, {1}, {0});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {3, 4, 5}, {1}, {2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {3, 4, 5}, {1}, {2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv16, data_types::i8, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {1}, {3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {1}, {3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_42x36x1x1_to_42x36x1x5_w_o_b_axes) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {42, 36, 1, 5}, {42, 36, 1, 1}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv166_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv166_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv326_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv326_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv166_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {1, 45, 2, 3}, {1, 45, 1, 3}, {0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv166_1x45x1x3_to_1x45x2x3_w_b_axes_0) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {1, 45, 2, 3}, {1, 45, 1, 3}, {0});
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3}, {2}, {0});
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2, 4}, {0, 2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2, 3}, {0, 1});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {4}, {2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {4}, {2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {3}, {1});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {3}, {1});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2}, {0});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2}, {0});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4, 5}, {0, 2, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 5}, {0, 1, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 4}, {0, 1, 2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv16, data_types::i8, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {4, 5}, {2, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {4, 5}, {2, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 5}, {1, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 5}, {1, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4}, {1, 2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4}, {1, 2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 5}, {0, 3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 5}, {0, 3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4}, {0, 2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4}, {0, 2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3}, {0, 1});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3}, {0, 1});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {5}, {3});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {5}, {3});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {4}, {2});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {4}, {2});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3}, {1});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3}, {1});
}


TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<ov::float16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2}, {0});
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    start_broadcast_test<ov::float16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2}, {0});
}


TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1x48x1x1_to_1x48x1x5_w_b_axes_0) {
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 1, 48, 1, 5 }, { 1, 48, 1, 1 }, { 0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1x48x1x1_to_1x48x1x5_w_b_axes_0) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 1, 48, 1, 5 }, { 1, 48, 1, 1 }, { 0, 1, 2, 3});
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1x48x1x1_to_1x48x1x5_w_b_axes_0) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 1, 48, 1, 5 }, { 1, 48, 1, 1 }, { 0, 1, 2, 3});
}

TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1x48x1x1_to_1x48x1x5_w_b_axes_0) {
    start_broadcast_test_5d<ov::float16>(format::b_fs_zyx_fsv16, data_types::f16, { 1, 48, 1, 5 }, { 1, 48, 1, 1 }, { 0, 1, 2, 3});
}

TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 4, 5 }, { 1 }, { 0 });
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 4, 5 }, { 1 }, { 0 });
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 4, 5 }, { 1 }, { 0 });
}

TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    start_broadcast_test_5d<ov::float16>(format::b_fs_zyx_fsv16, data_types::f16, { 4, 5 }, { 1 }, { 0 });
}


TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 2, 3, 4, 5, 2 }, { 1 }, { 2});
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 2, 3, 4, 5, 2 }, { 1 }, { 2});
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 2, 3, 4, 5, 2 }, { 1 }, { 2});
}

TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<ov::float16>(format::b_fs_zyx_fsv16, data_types::f16, { 2, 3, 4, 5, 2 }, { 1 }, { 2});
}

TEST(export_import_broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    start_broadcast_test_5d<ov::float16>(format::b_fs_zyx_fsv16, data_types::f16, { 2, 3, 4, 5, 2 }, { 1 }, { 2}, true);
}

static void run_broadcast_gpu_opt_y_axis(std::vector<ov::Dimension::value_type> in_static_shape,
                                    std::vector<ov::Dimension::value_type> in_dynamic_shape,
                                    std::vector<ov::Dimension::value_type> output_shape,
                                    std::vector<int32_t> target_shape) {

    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));


    auto input_static_layout = cldnn::layout{ov::PartialShape{in_static_shape}, data_types::f16, format::bfzyx};
    auto input_dynamic_layout = cldnn::layout{ov::PartialShape{in_dynamic_shape}, data_types::f16, format::bfzyx};
    auto target_shape_layout = cldnn::layout{ov::PartialShape{static_cast<ov::Dimension::value_type>(target_shape.size())},
                                                                data_types::i32, format::bfyx};

    primitive_id input_id = "input";
    primitive_id target_shape_id = "target_shape";

    topology topology;
    topology.add(input_layout(input_id, input_dynamic_layout));
    topology.add(input_layout(target_shape_id, target_shape_layout));
    auto broadcast_prim = broadcast("broadcast", input_info(input_id), input_info(target_shape_id), ov::AxisSet({}), ov::op::BroadcastType::BIDIRECTIONAL);
    if (output_shape.size() == 6) {
        broadcast_prim.output_pshape = ov::PartialShape({-1, -1, output_shape[2], output_shape[3], output_shape[4], output_shape[5]});
    } else if (output_shape.size() == 5) {
        broadcast_prim.output_pshape = ov::PartialShape({-1, -1, output_shape[2], output_shape[3], output_shape[4]});
    } else {
        broadcast_prim.output_pshape = ov::PartialShape({-1, -1, output_shape[2], output_shape[3]});
    }
    topology.add(broadcast_prim);
    topology.add(reorder("output", input_info("broadcast"), format::bfzyx, data_types::f16));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    size_t input_data_size = accumulate(in_static_shape.rbegin(), in_static_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(input_data_size, (size_t)1);
    std::vector<ov::float16> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((ov::float16)i);
    }
    auto input_mem = engine.allocate_memory(input_static_layout);
    set_values(input_mem, input_data);
    network->set_input_data(input_id, input_mem);

    auto target_shape_mem = engine.allocate_memory(target_shape_layout);
    set_values(target_shape_mem, target_shape);
    network->set_input_data(target_shape_id, target_shape_mem);

    auto outputs = network->execute();
    auto output = outputs.at("output").get_memory();

    ASSERT_NE(output, nullptr);
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());

    size_t output_data_size = accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>());
    ASSERT_GE(output_data_size, (size_t)1);
    std::vector<ov::float16> output_ref(output_data_size);
    ov::reference::broadcast(reinterpret_cast<const char*>(input_data.data()),
                             reinterpret_cast<char*>(output_ref.data()),
                             ov::Shape(in_static_shape.begin(), in_static_shape.end()),
                             ov::Shape(output_shape.begin(), output_shape.end()),
                             ov::AxisSet({}),
                             sizeof(ov::float16));

    ASSERT_EQ(output_ref.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));

    if (output_shape.size() == 6) {
        for (tensor::value_type b = 0; b < output_shape.at(0); ++b) {
            for (tensor::value_type f = 0; f < output_shape.at(1); ++f) {
                for (tensor::value_type w = 0; w < output_shape.at(2); ++w) {
                    for (tensor::value_type z = 0; z < output_shape.at(3); ++z) {
                        for (tensor::value_type y = 0; y < output_shape.at(4); ++y) {
                            for (tensor::value_type x = 0; x < output_shape.at(5); ++x) {
                                auto output_off = ((((b * output_shape.at(1) + f) * output_shape.at(2) + w) * output_shape.at(3) + z) * output_shape.at(4) + y) * output_shape.at(5) + x;
                                ASSERT_EQ(output_ptr[output_off], output_ref[output_off]);
                            }
                        }
                    }
                }
            }
        }
    } else if (output_shape.size() == 5) {
        for (tensor::value_type b = 0; b < output_shape.at(0); ++b) {
            for (tensor::value_type f = 0; f < output_shape.at(1); ++f) {
                for (tensor::value_type z = 0; z < output_shape.at(2); ++z) {
                    for (tensor::value_type y = 0; y < output_shape.at(3); ++y) {
                        for (tensor::value_type x = 0; x < output_shape.at(4); ++x) {
                            auto output_off = (((b * output_shape.at(1) + f) * output_shape.at(2) + z) * output_shape.at(3) + y) * output_shape.at(4) + x;
                            ASSERT_EQ(output_ptr[output_off], output_ref[output_off]);
                        }
                    }
                }
            }
        }
    } else {
        for (tensor::value_type b = 0; b < output_shape.at(0); ++b) {
            for (tensor::value_type f = 0; f < output_shape.at(1); ++f) {
                for (tensor::value_type y = 0; y < output_shape.at(2); ++y) {
                    for (tensor::value_type x = 0; x < output_shape.at(3); ++x) {
                        auto output_off = ((b * output_shape.at(1) + f) * output_shape.at(2) + y) * output_shape.at(3) + x;
                        ASSERT_EQ(output_ptr[output_off], output_ref[output_off]);
                    }
                }
            }
        }
    }
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x128_to_21x1x2x16x128_axis_Y_16) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,128},{-1,-1,2,1,128},{21,1,2,16,128},{1,1,1,16,1});
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x128_to_21x1x2x17x128_axis_Y_17) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,128},{-1,-1,2,1,128},{21,1,2,17,128},{1,1,1,17,1});
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x129_to_21x1x2x19x129_axis_Y_19) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,129},{-1,-1,2,1,129},{21,1,2,19,129},{1,1,1,19,1});
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x1_to_21x1x2x19x1_axis_Y_19) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,1},{-1,-1,2,1,1},{21,1,2,19,1},{1,1,1,19,1});
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x1_to_21x1x2x3x1_axis_Y_3) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,1},{-1,-1,2,1,1},{21,1,2,3,1},{1,1,1,3,1});
}

TEST(broadcast_gpu_opt_fp16, bfwzyx_21x1x2x1x1_to_21x1x2x19x1_axis_F_14_Z_12) {
    run_broadcast_gpu_opt_y_axis({21,14,12,1,1},{-1,14,12,1,1},{21,14,12,1,1},{1,14,12,1,1});
}

TEST(broadcast_gpu_opt_fp16, bfyx_21x2x1x1_to_21x2x3x1_axis_Y_13) {
    run_broadcast_gpu_opt_y_axis({21,2,1,5},{-1,2,1,5},{21,2,13,5},{1,1,13,1});
}

TEST(broadcast_gpu_opt_fp16, bfzyx_21x1x2x1x5_to_21x1x2x15x5_axis_Y_15) {
    run_broadcast_gpu_opt_y_axis({21,1,2,1,5},{-1,-1,2,1,5},{21,1,2,15,5},{1,1,1,15,1});
}

TEST(broadcast_gpu_opt_fp16, bfyx_4x5x1x1_to_4x5x6x1_axis_Y_6) {
    run_broadcast_gpu_opt_y_axis({4,5,1,1},{-1,-1,1,1},{4,5,6,1},{1,1,6,1});
}

TEST(broadcast_gpu_opt_fp16, bfyx_4x5x1x1_to_4x5x11x1_axis_Y_11) {
    run_broadcast_gpu_opt_y_axis({4,5,1,1},{-1,-1,1,1},{4,5,11,1},{1,1,11,1});
}
