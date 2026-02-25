// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/select.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "select_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct select_test_params {
    layout mask;
    layout input1_layout;
    layout input2_layout;
    ov::op::AutoBroadcastType broadcast_type;
    layout expected_layout;
};

class select_test : public testing::TestWithParam<select_test_params> { };

TEST_P(select_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_prim = std::make_shared<input_layout>("input0", p.mask);
    auto input1_prim = std::make_shared<input_layout>("input1", p.input1_layout);
    auto input2_prim = std::make_shared<input_layout>("input2", p.input2_layout);

    const ov::op::AutoBroadcastSpec& broadcast_spec = ov::op::AutoBroadcastSpec(p.broadcast_type);
    auto select_prim = std::make_shared<cldnn::select>("select_output", input_info("input0"), input_info("input1"), input_info("input2"), broadcast_spec);

    cldnn::program prog(engine);

    auto& input0_node = prog.get_or_create(input0_prim);
    auto& input1_node = prog.get_or_create(input1_prim);
    auto& input2_node = prog.get_or_create(input2_prim);

    auto& select_node = prog.get_or_create(select_prim);
    program_wrapper::add_connection(prog, input0_node, select_node);
    program_wrapper::add_connection(prog, input1_node, select_node);
    program_wrapper::add_connection(prog, input2_node, select_node);

    auto res = select_inst::calc_output_layouts<ov::PartialShape>(select_node, *select_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, select_test,
    testing::ValuesIn(std::vector<select_test_params>{
        {
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NONE,
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, 3, 4}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 1, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{   3, 4}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape{3, 3, 4}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, 1, 4}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 1, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{   5, 4}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape{3, 5, 4}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::PDPD,
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{      4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 3, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{          }, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::PDPD,
            layout{ov::PartialShape{2, 3, 4, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, -1}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NONE,
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, -1}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{-1, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, -1}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NONE,
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NONE,
            layout{ov::PartialShape{3, 2, 4}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NONE,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{-1, 1, -1}, data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 1, -1}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{   5, 4}, data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape{3, 5, 4}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}, 
            layout{ov::PartialShape{3, 2, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape{3, 2, 4}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::NUMPY,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{      4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 3, 4, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::PDPD,
            layout{ov::PartialShape{2, 3, 4, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            ov::op::AutoBroadcastType::PDPD,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
