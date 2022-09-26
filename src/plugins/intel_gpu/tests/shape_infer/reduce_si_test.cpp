// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "reduce_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct reduce_test_params {
    layout input;
    reduce_mode mode;
    std::vector<int64_t> axes;
    bool keep_dims;
    layout expected_layout;
};

class reduce_test : public testing::TestWithParam<reduce_test_params> { };

TEST_P(reduce_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.input);
    auto reduce_prim = std::make_shared<reduce>("output", "input", p.mode, p.axes, p.keep_dims);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& reduce_node = prog.get_or_create(reduce_prim);
    program_wrapper::add_connection(prog, input_node, reduce_node);
    auto res = reduce_inst::calc_output_layouts<ov::PartialShape>(reduce_node, *reduce_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, reduce_test,
    testing::ValuesIn(std::vector<reduce_test_params>{
        {
            layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx},
            reduce_mode::max, {1}, false,
            layout{ov::PartialShape{1}, data_types::f32, format::bfyx}
        },
            {
            layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx},
            reduce_mode::min, {1}, true,
            layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24}, data_types::f32, format::bfyx},
            reduce_mode::mean, {1}, false,
            layout{ov::PartialShape{6, 10, 24}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24}, data_types::f32, format::bfyx},
            reduce_mode::prod, {1}, true,
            layout{ov::PartialShape{6, 1, 10, 24}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12}, data_types::f32, format::bfyx},
            reduce_mode::sum, {1}, false,
            layout{ov::PartialShape{6}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10}, data_types::f32, format::bfyx},
            reduce_mode::logical_and, {1}, false,
            layout{ov::PartialShape{6, 10}, data_types::i8, format::bfyx}
        },
                {
            layout{ov::PartialShape{6, 12, 10}, data_types::f32, format::bfyx},
            reduce_mode::logical_or, {1}, true,
            layout{ov::PartialShape{6, 1, 10}, data_types::i8, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24}, data_types::f32, format::bfyx},
            reduce_mode::l1, {2, 3}, false,
            layout{ov::PartialShape{6, 12}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24}, data_types::f32, format::bfyx},
            reduce_mode::l2, {2, 3}, true,
            layout{ov::PartialShape{6, 12, 1, 1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24}, data_types::f32, format::bfyx},
            reduce_mode::max, {-2}, false,
            layout{ov::PartialShape{6, 12, 24}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24, 3}, data_types::f32, format::bfzyx},
            reduce_mode::min, {2, 3}, false,
            layout{ov::PartialShape{6, 12, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{6, 12, 10, 24, 3, 5}, data_types::f32, format::bfwzyx},
            reduce_mode::sum, {2, 3}, false,
            layout{ov::PartialShape{6, 12, 3, 5}, data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
