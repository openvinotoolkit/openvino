// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "softmax_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace memory_realloc_tests {

TEST(softmax_gpu_dynamic_f32_test_upper_bound, input_same_values) {
    static const int32_t
        output_x_1  = 10, output_b_1  = 8,
        input_x_1   = 10, input_b_1   = 8,
        out_size_1  = output_x_1 * output_b_1,
        output_x_2  = 10, output_b_2  = 4,
        input_x_2   = 10, input_b_2  = 4,
        out_size_2  = output_x_2 * output_b_2;

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
    network network(engine, topology(input_layout("input", in_layout), softmax("softmax", input_info("input"), 3)));

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
    }
}
}  // memory_realloc_tests
