// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "random_uniform_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct random_uniform_si_test_params {
    ov::PartialShape expected_out_pshape;
    data_types out_data_type;
    std::pair<float, float> min_max_vals;
};

class random_uniform_si_test : public testing::TestWithParam<random_uniform_si_test_params> { };

TEST_P(random_uniform_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    std::vector<layout> input_layouts;
    const size_t num_inputs = 3;

    for (size_t idx = 0; idx < num_inputs; idx++) {
        auto in_layout = layout{{1}, p.out_data_type, format::bfyx};
        if (idx == 0) {
            auto input_pshape = ov::PartialShape{static_cast<long int>(p.expected_out_pshape.size())};
            in_layout = layout{input_pshape, data_types::i64, format::bfyx};
        }
        input_layouts.push_back(in_layout);

        auto prim_id = "input_" + std::to_string(idx);
        auto const_data_prim = std::make_shared<input_layout>(prim_id, in_layout);
        input_prims.push_back(const_data_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto random_uniform_prim = std::make_shared<random_uniform>("random_uniform", input_prim_ids, p.out_data_type, 0, 0);
    auto& random_uniform_node = prog.get_or_create(random_uniform_prim);

    for (auto& iprim : input_prims) {
        auto& input_node = prog.get_or_create(iprim);
        program_wrapper::add_connection(prog, input_node, random_uniform_node);
    }

    auto params = random_uniform_node.get_kernel_impl_params();
    params->memory_deps.clear();
    auto get_mem = [&](size_t idx, float val) -> memory::ptr {
        auto in_layout = input_layouts[idx];
        auto allocated_mem = engine.allocate_memory(in_layout);
        switch (p.out_data_type) {
            case data_types::f16:
                set_values(allocated_mem, {ov::float16(val).to_bits()});
                break;
            case data_types::f32:
                set_values(allocated_mem, {static_cast<data_type_to_type<data_types::f32>::type>(val)});
                break;
            case data_types::i32:
                set_values(allocated_mem, {static_cast<data_type_to_type<data_types::i32>::type>(val)});
                break;
            case data_types::i64:
                set_values(allocated_mem, {static_cast<data_type_to_type<data_types::i64>::type>(val)});
                break;
            case data_types::i8:
                set_values(allocated_mem, {static_cast<data_type_to_type<data_types::i8>::type>(val)});
                break;
            case data_types::u8:
                set_values(allocated_mem, {static_cast<data_type_to_type<data_types::u8>::type>(val)});
                break;
            default:
                break;
        }
        return allocated_mem;
    };

    if (p.expected_out_pshape.is_static()) {
        auto input_mem = engine.allocate_memory(input_layouts[0]);
        set_values(input_mem, p.expected_out_pshape.get_shape());
        params->memory_deps.emplace(0, input_mem);
    }

    params->memory_deps.emplace(1, get_mem(1, p.min_max_vals.first));
    params->memory_deps.emplace(2, get_mem(2, p.min_max_vals.second));

    if (p.min_max_vals.first < p.min_max_vals.second) {
        auto res = random_uniform_inst::calc_output_layouts<ov::PartialShape>(random_uniform_node, *params);

        auto expected_out_layout = layout{p.expected_out_pshape, p.out_data_type, format::get_default_format(p.expected_out_pshape.size())};
        ASSERT_EQ(res.size(), 1);
        ASSERT_EQ(res[0], expected_out_layout);
    } else {
        ASSERT_ANY_THROW(random_uniform_inst::calc_output_layouts<ov::PartialShape>(random_uniform_node, *params));
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, random_uniform_si_test,
    testing::ValuesIn(std::vector<random_uniform_si_test_params>{
        {ov::PartialShape{2}, data_types::i32, {0, 10}},
        {ov::PartialShape{2}, data_types::i8,  {0, 10}},
        {ov::PartialShape{2}, data_types::u8,  {0, 10}},
        {ov::PartialShape{2}, data_types::i64, {0, 10}},
        {ov::PartialShape{2}, data_types::i32, {0, 10}},
        {ov::PartialShape{2}, data_types::f32, {0, 10}},
        {ov::PartialShape{2}, data_types::f16, {0, 10}},
        {ov::PartialShape{2,4}, data_types::i32, {0, 10}},
        {ov::PartialShape{2,4}, data_types::f32, {0, 10}},
        {ov::PartialShape{2,4,3}, data_types::i32, {0, 10}},
        {ov::PartialShape{2,4,3}, data_types::f32, {0, 10}},
        {ov::PartialShape{2,4,3,2}, data_types::i32, {0, 10}},
        {ov::PartialShape{2,4,3,2}, data_types::f32, {0, 10}},
        {ov::PartialShape{2,4,3,1,2}, data_types::i32, {0, 10}},
        {ov::PartialShape{2,4,3,1,2}, data_types::f32, {0, 10}},

        // Dynamic output shape
        {ov::PartialShape::dynamic(1), data_types::f32, {0, 10}},
        {ov::PartialShape::dynamic(2), data_types::f32, {0, 10}},
        {ov::PartialShape::dynamic(3), data_types::f32, {0, 10}},
        {ov::PartialShape::dynamic(4), data_types::f32, {0, 10}},
        {ov::PartialShape::dynamic(5), data_types::f32, {0, 10}},

        // Incorrect min/max values
        {ov::PartialShape{2}, data_types::i32, {20, 20}},
        {ov::PartialShape{2,4,3,1,2}, data_types::i32, {20, 10}},
        {ov::PartialShape::dynamic(1), data_types::f32, {20, 20}},
        {ov::PartialShape::dynamic(5), data_types::f32, {20, 10}},
    }));

};  // shape_infer_tests
