// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "crop_inst.h"
#include "concatenation_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct crop_si_test_params {
    tensor reference_input_size;
    std::vector<tensor> offsets;
    std::vector<std::vector<int64_t>> const_values;
    int64_t axis;
    std::vector<layout> input_layouts;
    std::vector<layout> expected_layouts;
    size_t param_num_splits;
};

class crop_si_test : public testing::TestWithParam<crop_si_test_params> { };

TEST_P(crop_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    {
        auto prim_id = "data0";
        auto data_layout_prim = std::make_shared<input_layout>(prim_id, p.input_layouts[0]);
        input_prims.push_back(data_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    for (size_t i = 1; i < p.input_layouts.size(); i++) {
        auto prim_id = "const::data"+std::to_string(i);
        auto prim_mem = engine.allocate_memory(p.input_layouts[i]);
        set_values(prim_mem, p.const_values[i-1]);
        auto const_data_prim = std::make_shared<data>(prim_id, prim_mem);
        input_prims.push_back(const_data_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    crop_ngraph_op_mode op_mode = crop_ngraph_op_mode::none;
    if (p.const_values.size() == 2) {
        op_mode = crop_ngraph_op_mode::variadic_split;
    } else if (p.const_values.size() == 1) {
        op_mode = crop_ngraph_op_mode::split;
    }

    for (size_t output_idx = 0; output_idx < p.expected_layouts.size(); output_idx++) {
        auto prim_id = "crop.out" + std::to_string(output_idx);
        auto crop_prim = std::make_shared<crop>(prim_id, input_prim_ids, p.reference_input_size, p.offsets[output_idx], op_mode, static_cast<int>(output_idx), p.axis, p.param_num_splits);
        auto& crop_node = prog.get_or_create(crop_prim);

        for (auto& prim : input_prims) {
            auto& input_node = prog.get_or_create(prim);
            program_wrapper::add_connection(prog, input_node, crop_node);
        }

        auto params = crop_node.get_kernel_impl_params();
        auto res = crop_inst::calc_output_layouts<ov::PartialShape>(crop_node, *params);

        ASSERT_EQ(res.size(), 1);
        ASSERT_EQ(res[0], p.expected_layouts[output_idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, crop_si_test,
    testing::ValuesIn(std::vector<crop_si_test_params>{
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {{-1}, {1,1}},
            -1,
            {{{1,32,2},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}, {{2},data_types::i64,format::bfyx}},
            {{{1,32,1},data_types::f32,format::bfyx}, {{1,32,1},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {{-1}, {1,1}},
            -1,
            {{ov::PartialShape::dynamic(),data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}, {{2},data_types::i64,format::bfyx}},
            {{ov::PartialShape::dynamic(),data_types::f32,format::bfyx}, {ov::PartialShape::dynamic(),data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({3,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {},
            -1,
            {{ov::PartialShape::dynamic(),data_types::f32,format::bfyx}},
            {{ov::PartialShape::dynamic(1),data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({3,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {},
            -1,
            {{{4},data_types::f32,format::bfyx}},
            {{{3},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({-1,-1,-1,-1,-1,-1,-1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {},
            -1,
            {{{4,3,2,5},data_types::f32,format::bfyx}},
            {{{3,2,1,4},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,1,0,0,1,1,1}),tensor({0,2,0,0,1,1,1}),tensor({0,3,0,0,1,1,1})},
            {{1}, {1,1,1,1}},
            1,
            {{{4819,4,1,1,4},data_types::f32,format::bfzyx}, {{},data_types::i64,format::bfzyx}, {{4},data_types::i64,format::bfzyx}},
            {{{4819,1,1,1,4},data_types::f32,format::bfzyx}, {{4819,1,1,1,4},data_types::f32,format::bfzyx}, {{4819,1,1,1,4},data_types::f32,format::bfzyx}, {{4819,1,1,1,4},data_types::f32,format::bfzyx}}, 0
        },
        {
            tensor({4507,1,1,1,1,1,1}),
            {tensor({0,2,0,0,1,1,1})},
            {},
            -1,
            {{{4507,3,1,1},data_types::f32,format::bfyx}},
            {{{4507,1,1,1},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,0,0,0,1,1,1})},
            {{2}, {11,3}},
            2,
            {{{1,14,14,384},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}, {{2},data_types::i64,format::bfyx}},
            {{{1,14,11,384},data_types::f32,format::bfyx}, {{1,14,3,384},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,2048,1,1,1,1}),
            {tensor({0,2,0,0,1,1,1})},
            {},
            -1,
            {{{1,16,1,2048},data_types::f32,format::bfyx}},
            {{{1,1,1,2048},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}),tensor({0,1320,0,0,1,1,1})},
            {{1},{1320,99}},
            1,
            {{{1,1419},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}, {{2},data_types::i64,format::bfyx}},
            {{{1,1320},data_types::f32,format::bfyx}, {{1,99},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,128,2,64,1,1,1}),
            {tensor({0,0,8,0,1,1,1})},
            {},
            -1,
            {{{1,128,64,10},data_types::f32,format::bfyx}},
            {{{1,128,64,2},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}), tensor({0,1,0,0,1,1,1})},
            {{1}},
            1,
            {{{4,2},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}},
            {{{4,1},data_types::f32,format::bfyx}, {{4,1},data_types::f32,format::bfyx}}, 2
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1}), tensor({0,0,2048,0,1,1,1})},
            {{2}},
            2,
            {{{5,1,4096,1},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}},
            {{{5,1,2048,1},data_types::f32,format::bfyx}, {{5,1,2048,1},data_types::f32,format::bfyx}}, 2
        },
        {
            tensor({1,1400,1,1,1,1,1}),
            {tensor({0,100,0,0,1,1,1})},
            {},
            -1,
            {{{1,1500,1,1},data_types::f32,format::bfyx}},
            {{{1,1400,1,1},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({1,1,1,1,1,1,1}),
            {tensor({0,0,0,0,1,1,1})},
            {{2},{1}},
            2,
            {{{7,1,1},data_types::f32,format::bfyx}, {{},data_types::i64,format::bfyx}, {{1},data_types::i64,format::bfyx}},
            {{{7,1,1},data_types::f32,format::bfyx}}, 0
        },
        {
            tensor({128,100,1,3,1,1,1}),
            {tensor({0,0,0,0,1,1,1})},
            {},
            -1,
            {{{128,100,4},data_types::f32,format::bfyx}},
            {{{128,100,3},data_types::f32,format::bfyx}}, 0
        }
    }));

};  // shape_infer_tests
