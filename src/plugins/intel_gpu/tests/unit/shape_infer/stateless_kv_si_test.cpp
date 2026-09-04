// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/stateless_kv.hpp>

#include "stateless_kv_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct stateless_kv_test_params {
    std::vector<layout> input_layouts;
    int64_t concat_axis;
    bool is_present_len;
    std::vector<layout> expected_layouts;
};

class stateless_kv_test : public testing::TestWithParam<stateless_kv_test_params> {};

TEST_P(stateless_kv_test, shape_infer) {
    const auto& p = GetParam();

    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<input_layout>> input_prims;
    std::vector<input_info> input_prim_ids;
    for (size_t i = 0; i < p.input_layouts.size(); ++i) {
        const auto prim_id = "data" + std::to_string(i);
        input_prims.push_back(std::make_shared<input_layout>(prim_id, p.input_layouts[i]));
        input_prim_ids.emplace_back(prim_id);
    }

    auto stateless_kv_prim =
        std::make_shared<stateless_kv>("output", input_prim_ids, p.concat_axis, p.is_present_len);
    stateless_kv_prim->num_outputs = 2;
    stateless_kv_prim->output_data_types = {p.input_layouts[0].data_type, p.input_layouts[0].data_type};
    auto& stateless_kv_node = prog.get_or_create(stateless_kv_prim);
    for (const auto& input_prim : input_prims) {
        auto& input_node = prog.get_or_create(input_prim);
        program_wrapper::add_connection(prog, input_node, stateless_kv_node);
    }

    auto params = stateless_kv_node.get_kernel_impl_params();
    const auto result = stateless_kv_inst::calc_output_layouts<ov::PartialShape>(stateless_kv_node, *params);

    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result, p.expected_layouts);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         stateless_kv_test,
                         testing::ValuesIn(std::vector<stateless_kv_test_params>{
                             {
                                 {
                                     layout{ov::PartialShape{1, 2, 16, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{1, 2, 1, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                 },
                                 2,
                                 true,
                                 {
                                     layout{ov::PartialShape{1, 2, 16, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{1, 2, -1, 4}, data_types::f16, format::bfyx, padding{{}, {}, padding::DynamicDimsMask{"0100"}}},
                                 },
                             },
                             {
                                 {
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{-1, 2, 1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                 },
                                 2,
                                 false,
                                 {
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx, padding{{}, {}, padding::DynamicDimsMask{"0100"}}},
                                 },
                             },
                             {
                                 {
                                     layout{ov::PartialShape{1, 2, 16, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{1, 2, 16, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                 },
                                 0,
                                 true,
                                 {
                                     layout{ov::PartialShape{1, 2, 16, 4}, data_types::f16, format::bfyx},
                                     layout{ov::PartialShape{-1, 2, 16, 4}, data_types::f16, format::bfyx, padding{{}, {}, padding::DynamicDimsMask{"0001"}}},
                                 },
                             },
                             {
                                 {
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{-1, 2, 1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                     layout{ov::PartialShape{1}, data_types::i64, format::bfyx},
                                 },
                                 -2,
                                 false,
                                 {
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
                                     layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx, padding{{}, {}, padding::DynamicDimsMask{"0100"}}},
                                 },
                             },
                         }));

}  // shape_infer_tests
