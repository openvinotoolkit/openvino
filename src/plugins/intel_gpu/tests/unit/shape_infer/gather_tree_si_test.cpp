// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_tree.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gather_tree_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct gather_tree_test_params {
    layout step_ids;
    layout parent_ids;
    layout max_seq_len;
    layout end_token;
    layout expected_layout;
};

class gather_tree_test : public testing::TestWithParam<gather_tree_test_params> { };

TEST_P(gather_tree_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_prim = std::make_shared<input_layout>("input0", p.step_ids);
    auto input1_prim = std::make_shared<input_layout>("input1", p.parent_ids);
    auto input2_prim = std::make_shared<input_layout>("input2", p.max_seq_len);
    auto input3_prim = std::make_shared<input_layout>("input3", p.end_token);
    auto gather_prim = std::make_shared<gather_tree>("output",
                                                    input_info("input0"),
                                                    input_info("input1"),
                                                    input_info("input2"),
                                                    input_info("input3"));

    cldnn::program prog(engine);

    auto& input0_node = prog.get_or_create(input0_prim);
    auto& input1_node = prog.get_or_create(input1_prim);
    auto& input2_node = prog.get_or_create(input2_prim);
    auto& input3_node = prog.get_or_create(input3_prim);

    auto& gather_tree_node = prog.get_or_create(gather_prim);
    program_wrapper::add_connection(prog, input0_node, gather_tree_node);
    program_wrapper::add_connection(prog, input1_node, gather_tree_node);
    program_wrapper::add_connection(prog, input2_node, gather_tree_node);
    program_wrapper::add_connection(prog, input3_node, gather_tree_node);

    auto res = gather_tree_inst::calc_output_layouts<ov::PartialShape>(gather_tree_node, *gather_tree_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gather_tree_test,
    testing::ValuesIn(std::vector<gather_tree_test_params>{
        {
            layout{ov::PartialShape{100, 1, 10}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{100, 1, 10}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{     1    }, data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{100, 1, 10}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{    4   }, data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{    4   }, data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{    4   }, data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{20, 4, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{    4    }, data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{-1, 4, -1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(0), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        },
    }));

}  // namespace shape_infer_tests
