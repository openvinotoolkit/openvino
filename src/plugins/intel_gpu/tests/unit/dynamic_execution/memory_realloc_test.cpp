// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "softmax_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace memory_realloc_tests {
TEST(memory_reuse_realloc_reset_test, basic_conv_with_padding) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx};

    auto weights = engine.allocate_memory(weight_layout);
    set_values<ov::float16>(weights, {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            //
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
    });

    layout input_layout_1 = layout{ov::PartialShape{1, 3, 5, 5}, data_types::f32, format::bfyx};
    auto input_mem_1 = engine.allocate_memory(input_layout_1);
    set_values(input_mem_1, {
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        });

    std::vector<float> ref_output_1 = {6,   18,  36, 54,  72,  54,  30,  12,  36, 72, 108, 144, 108,
                                       60,  18,  54, 108, 162, 216, 162, 90,  18, 54, 108, 162, 216,
                                       162, 90,  18, 54,  108, 162, 216, 162, 90, 12, 36,  72,  108,
                                       144, 108, 60, 6,   18,  36,  54,  72,  54, 30};

    layout input_layout_2 = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem_2 = engine.allocate_memory(input_layout_2);
    set_values(input_mem_2, {11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref_output_2 = { 66, 132, 132, 66, 132, 264, 264, 132, 132, 264, 264, 132, 66, 132, 132, 66};
     std::vector<float> values_to_subtract = {};
    auto input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weights", weights),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("reorder"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("output", input_info("conv"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", {format::any, "", impl_types::ocl}}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem_1);
    auto outputs_1 = network.execute();
    network.set_input_data("input", input_mem_2);
    auto outputs_2 = network.execute();
    auto output_mem_2 = outputs_2.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_2_ptr(output_mem_2, get_test_stream());
    for (size_t i = 0; i < output_mem_2->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_2_ptr[i], ref_output_2[i]);
    }
    // check padding of second run of reorder
    // 0, 0, 0,  0,  0, 0,
    // 0, 0, 0,  0,  0, 0,
    // 0, 0, 11, 11, 0, 0,
    // 0, 0, 11, 11, 0, 0,
    // 0, 0,"0","0","0","0", // !! check pad_after
    // 0, 0,"0","0","0","0", // !! check pad_after
    auto reorder_mem = network.get_primitive("reorder")->output_memory_ptr();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> reorder_mem_ptr(reorder_mem, get_test_stream());
    for (size_t i = 26; i < 29; ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    for (size_t i = 32; i < 35; ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    // Mem should be reallocate when request size is bigger than existing buffer size
    ASSERT_TRUE(reorder_mem->size() <= reorder_mem->get_mem_tracker()->size())
                << "reorder mem buffer size: " <<  reorder_mem->size() << "bytes is bigger than original size of allocated mem: "
                << reorder_mem->get_mem_tracker()->size() << "bytes.";
}

TEST(softmax_gpu_dynamic_f32_test_upper_bound, input_same_values) {
    static const int32_t
        output_x_1  = 10, output_b_1  = 8,
        input_x_1   = 10, input_b_1   = 8,
        out_size_1  = output_x_1 * output_b_1,
        output_x_2  = 10, output_b_2  = 4,
        input_x_2   = 10, input_b_2  = 4,
        out_size_2  = output_x_2 * output_b_2,
        output_x_3  = 10, output_b_3  = 16,
        input_x_3   = 10, input_b_3  = 16,
        out_size_3  = output_x_3 * output_b_3;

    cldnn::engine& engine = get_test_engine();

    auto compare_out_buffer_with_expected = [&](float* out_buffer, std::vector<float>& expected_buffer, size_t size) {
        for(size_t i = 0; i < size; ++i) {
            // does output have expected values
            ASSERT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    };
    auto in_layout =
        layout(ov::PartialShape{ov::Dimension{1, 10}, ov::Dimension{1, 10}, ov::Dimension{1, 10}, ov::Dimension{1, 10}},
               data_types::f32,
               format::bfyx);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc softmax_impl = { format::bfyx, "softmax_gpu_ref" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "softmax", softmax_impl } }));
    network network(engine, topology(input_layout("input", in_layout),
                                     reorder("reorder", input_info("input"), format::bfyx, data_types::f16),
                                     softmax("softmax", input_info("reorder"), 3),
                                     reorder("reorder2", input_info("softmax"), format::bfyx, data_types::f32)),
                                     config);

    // First run
    float out_buffer_1[out_size_1];
    std::vector<float> in_b_1(out_size_1, 1.0f);
    std::vector<float> expected_buffer_1(out_size_1, 0.1f);
    cldnn::memory::ptr input_1 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_1, 1, input_x_1, 1}});
    set_values(input_1, in_b_1);
    network.set_input_data("input", input_1);

    auto outputs_1 = network.execute();
    auto output_mem_1 = outputs_1.begin()->second.get_memory();
    auto internal_mems_1 = network.get_primitive("softmax")->get_intermediates_memories();
    cldnn::mem_lock<float> output_ptr_1(output_mem_1, get_test_stream());
    for (uint32_t i = 0; i < out_size_1; i++) {
        out_buffer_1[i] = output_ptr_1[i];
    }
    compare_out_buffer_with_expected(out_buffer_1, expected_buffer_1, out_size_1);

    // Second run
    float out_buffer_2[out_size_2];
    std::vector<float> in_b_2(out_size_2, 2.0f);
    std::vector<float> expected_buffer_2(out_size_2, 0.1f);
    cldnn::memory::ptr input_2 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_2, 1, input_x_2, 1}});
    set_values(input_2, in_b_2);
    network.set_input_data("input", input_2);
    auto outputs_2 = network.execute();
    auto output_mem_2 = outputs_2.begin()->second.get_memory();
    auto internal_mems_2 = network.get_primitive("softmax")->get_intermediates_memories();
    cldnn::mem_lock<float> output_ptr_2(output_mem_2, get_test_stream());
    for (uint32_t i = 0; i < out_size_2; i++) {
        out_buffer_2[i] = output_ptr_2[i];
    }
    compare_out_buffer_with_expected(out_buffer_2, expected_buffer_2, out_size_2);

    // Check output is not reallocated
    ASSERT_EQ(output_ptr_1.data(), output_ptr_2.data());
    ASSERT_EQ(internal_mems_1.size(), internal_mems_2.size());
    for (size_t i = 0; i < internal_mems_1.size(); ++i) {
        ASSERT_EQ(internal_mems_1[i]->buffer_ptr(), internal_mems_2[i]->buffer_ptr());
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(internal_mems_1[i]->get_allocation_type(), allocation_type::usm_device);
        }
    }
    // Third run
    float out_buffer_3[out_size_3];
    std::vector<float> in_b_3(out_size_3, 2.0f);
    std::vector<float> expected_buffer_3(out_size_3, 0.1f);
    cldnn::memory::ptr input_3 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_3, 1, input_x_3, 1}});
    set_values(input_3, in_b_3);
    network.set_input_data("input", input_3);
    auto outputs_3 = network.execute();
    auto output_mem_3 = outputs_3.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_mem_3, get_test_stream());
    for (uint32_t i = 0; i < out_size_3; i++) {
        out_buffer_3[i] = output_ptr_3[i];
    }
    compare_out_buffer_with_expected(out_buffer_3, expected_buffer_3, out_size_3);
    auto internal_mems_3 = network.get_primitive("softmax")->get_intermediates_memories();
    for (size_t i = 0; i < internal_mems_3.size(); ++i) {
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(internal_mems_3[i]->get_allocation_type(), allocation_type::usm_device);
        }
    }
    auto& pool = network.get_memory_pool();
    // check if previously allocated internal buffer is released
    ASSERT_EQ(pool.get_non_padded_pool_size(), 3);
}

TEST(dyn_shape_mem_test, igpu_shape_infer_dep_mem_type) {
    auto& engine = get_test_engine();
    auto input_lay_1 = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    auto input_lay_2 = layout{ov::PartialShape::dynamic(2), data_types::i32, format::bfyx};
    topology topology(input_layout("input1", input_lay_1),
                      input_layout("pattern1", input_lay_2),
                      input_layout("pattern2", input_lay_2),
                      reorder("reorder", input_info("input1"), format::bfyx, data_types::f16),
                      eltwise("eltwise", {input_info("pattern1"), input_info("pattern2")}, eltwise_mode::sum, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)),
                      reshape("reshape", input_info("reorder"), input_info("eltwise"), false, ov::PartialShape()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{6, 1}, data_types::f32, format::bfyx});
    set_values<float>(input_mem, {11.f, 22.f, 33.f, 44.f, 55.f, 66.f});;
    auto pattern_mem1 = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i32, format::bfyx});
    set_values<int32_t>(pattern_mem1, {2, 1, 1, 0});;
    auto pattern_mem2 = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i32, format::bfyx});
    set_values<int32_t>(pattern_mem2, {1, 1, 0, 1});;

    network.set_input_data("input1", input_mem);
    network.set_input_data("pattern1", pattern_mem1);
    network.set_input_data("pattern2", pattern_mem2);
    auto output = network.execute();
    const auto& reorder_mem = network.get_primitive("reorder")->output_memory();
    const auto& pattern_mem = network.get_primitive("eltwise")->output_memory();
    ASSERT_EQ(reorder_mem.get_allocation_type(), allocation_type::usm_device);
    if (engine.get_device_info().dev_type == device_type::integrated_gpu) {
        // for iGPU, allocating shape infer dep mem to usm_host improves shape_infer performance by preventing memcpy b/w device to host mem
        ASSERT_EQ(pattern_mem.get_allocation_type(), allocation_type::usm_host);
    } else {
        // if allocate shape infer dep mem to host && write result from device && read from host, cache coherence issue occurs
        ASSERT_EQ(pattern_mem.get_allocation_type(), allocation_type::usm_device);
    }
    auto expected_layout = layout{ov::PartialShape{3, 2, 1, 1}, data_types::f16, format::bfyx};
    ASSERT_EQ(output.begin()->second.get_memory()->get_layout(), expected_layout);
}

TEST(memory_reuse_realloc_reset_test, basic_conv_with_padding_reorder) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx};

    auto weights = engine.allocate_memory(weight_layout);
    set_values<ov::float16>(weights, {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            //
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
    });

    layout input_layout_1 = layout{ov::PartialShape{1, 3, 5, 5}, data_types::f32, format::bfyx};
    auto input_mem_1 = engine.allocate_memory(input_layout_1);
    set_values(input_mem_1, {
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        });

    std::vector<float> ref_output_1 = {6,   18,  36, 54,  72,  54,  30,  12,  36, 72, 108, 144, 108,
                                       60,  18,  54, 108, 162, 216, 162, 90,  18, 54, 108, 162, 216,
                                       162, 90,  18, 54,  108, 162, 216, 162, 90, 12, 36,  72,  108,
                                       144, 108, 60, 6,   18,  36,  54,  72,  54, 30};

    layout input_layout_2 = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem_2 = engine.allocate_memory(input_layout_2);
    set_values(input_mem_2, {11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref_output_2 = { 66, 132, 132, 66, 132, 264, 264, 132, 132, 264, 264, 132, 66, 132, 132, 66};
     std::vector<float> values_to_subtract = {};
    auto input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weights", weights),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("reorder"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("output", input_info("conv"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", {format::any, "", impl_types::ocl}}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem_2);
    auto outputs_1 = network.execute();
    network.set_input_data("input", input_mem_1);
    auto outputs_2 = network.execute();

    // check padding of second run of reorder
    // 0, 0, 0, ... 0,  0, 0,
    // 0, 0, 0, ... 0,  0, 0,
    // 0, 0, 1, ... 5,  0, 0,
    // .  .   .
    // 0, 0, 1, ... 5,  0, 0,
    // 0, 0,"0", .. "0","0","0", // !! check pad_after
    // 0, 0,"0", .. "0","0","0", // !! check pad_after
    auto reorder_mem = network.get_primitive("reorder")->output_memory_ptr();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> reorder_mem_ptr(reorder_mem, get_test_stream());
    for (size_t i = (63 + 81 * 2); i < (71 + 81 * 2); ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    for (size_t i = (72 + 81 * 2); i < (80 + 81 * 2); ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    // Mem should be reallocate when request size is bigger than existing buffer size
    ASSERT_TRUE(reorder_mem->size() <= reorder_mem->get_mem_tracker()->size())
                << "reorder mem buffer size: " <<  reorder_mem->size() << "bytes is bigger than original size of allocated mem: "
                << reorder_mem->get_mem_tracker()->size() << "bytes.";
}
}  // memory_realloc_tests
