// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/pooling.hpp>

#include "pooling_inst.h"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct pooling_test_params {
    layout data_layout;
    pooling_mode mode;
    ov::Shape kernel_size;
    ov::Strides stride;
    ov::Shape pads_begin;
    ov::Shape pads_end;
    ov::op::PadType auto_pad;
    layout expected_layout;
};

class pooling_si_test : public testing::TestWithParam<pooling_test_params> { };

TEST_P(pooling_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto pooling_prim = std::make_shared<pooling>("output", input_info("data"), p.mode, p.kernel_size, p.stride, p.pads_begin, p.pads_end, p.auto_pad);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& pooling_node = prog.get_or_create(pooling_prim);
    program_wrapper::add_connection(prog, data_node, pooling_node);

    auto res = pooling_inst::calc_output_layouts<ov::PartialShape>(pooling_node, *pooling_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, pooling_si_test,
    testing::ValuesIn(std::vector<pooling_test_params>{
        {
            layout{ov::PartialShape{1, 3, 32, 32}, data_types::f32, format::bfyx},
            pooling_mode::max, {2, 2}, {2, 2}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT,
            layout{ov::PartialShape{1, 3, 17, 17}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 3, 32, 32}, data_types::f32, format::bfyx},
            pooling_mode::average, {5, 5}, {2, 2}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT,
            layout{ov::PartialShape{1, 3, 15, 15}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 3, 32, 32}, data_types::f32, format::bfyx},
            pooling_mode::average_no_padding, {5, 5}, {3, 3}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT,
            layout{ov::PartialShape{1, 3, 10, 10}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 3, 32, 32}, data_types::f32, format::bfyx},
            pooling_mode::average, {5, 5}, {2, 2}, {0, 0}, {1, 1}, ov::op::PadType::SAME_UPPER,
            layout{ov::PartialShape{1, 3, 16, 16}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 3, 32, 32}, data_types::f32, format::bfyx},
            pooling_mode::max, {2, 2}, {2, 2}, {1, 1}, {1, 1}, ov::op::PadType::VALID,
            layout{ov::PartialShape{1, 3, 16, 16}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, -1, -1, 32}, data_types::f32, format::bfyx},
            pooling_mode::max, {2, 2}, {2, 2}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT,
            layout{ov::PartialShape{1, -1, -1, -1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            pooling_mode::max, {2, 2}, {2, 2}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
