// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/ctc_greedy_decoder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "ctc_greedy_decoder_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct ctc_greedy_decoder_seq_len_test_params {
    std::vector<layout> in_layouts;
    std::vector<int32_t> blank_index;
    std::vector<layout> expected_layouts;
};

class ctc_greedy_decoder_seq_len_test : public testing::TestWithParam<ctc_greedy_decoder_seq_len_test_params> { };

TEST_P(ctc_greedy_decoder_seq_len_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    {
        auto prim_id = "input";
        auto input_layout_prim = std::make_shared<input_layout>(prim_id, p.in_layouts[0]);
        input_prims.push_back(input_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    for (size_t i = 1; i < p.in_layouts.size(); i++) {
        auto prim_id = "const" + std::to_string(i);
        auto prim_mem = engine.allocate_memory(p.in_layouts[i]);
        if (i == 2)
            set_values(prim_mem, p.blank_index);
        auto const_data_prim = std::make_shared<data>(prim_id, prim_mem);
        input_prims.push_back(const_data_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto ctc_greedy_decoder_seq_len_prim = std::make_shared<ctc_greedy_decoder>(
                                   "output",
                                   input_prim_ids,
                                   p.blank_index[0],
                                   true,
                                   data_types::i32,
                                   2);

    cldnn::program prog(engine);
    auto& ctc_greedy_decoder_seq_len_node = prog.get_or_create(ctc_greedy_decoder_seq_len_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, ctc_greedy_decoder_seq_len_node);
    }

    auto res = ctc_greedy_decoder_inst::calc_output_layouts<ov::PartialShape>(ctc_greedy_decoder_seq_len_node, *ctc_greedy_decoder_seq_len_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 2);
    for (size_t i = 0; i < p.expected_layouts.size(); i++)
        ASSERT_EQ(res[i], p.expected_layouts[i]);
}

INSTANTIATE_TEST_SUITE_P(smoke, ctc_greedy_decoder_seq_len_test,
    testing::ValuesIn(std::vector<ctc_greedy_decoder_seq_len_test_params>{
        {
            {
                {layout{ov::PartialShape{1, 6, 10}, data_types::f32, format::bfyx}},
                {layout{ov::PartialShape{1}, data_types::i32, format::bfyx}},
            },
            {-1},
            {
                {layout{ov::PartialShape{1, 6}, data_types::i32, format::bfyx}},
                {layout{ov::PartialShape{1}, data_types::i32, format::bfyx}},
            },
        },
        {
            {
                {layout{ov::PartialShape{1, 6, 10}, data_types::f32, format::bfyx}},
                {layout{ov::PartialShape{1}, data_types::i32, format::bfyx}},
                {layout{ov::PartialShape{1}, data_types::i32, format::bfyx}},
            },
            {5},
            {
                {layout{ov::PartialShape{1, 6}, data_types::i32, format::bfyx}},
                {layout{ov::PartialShape{1}, data_types::i32, format::bfyx}},
            },
        },
    }));

}  // namespace shape_infer_tests
