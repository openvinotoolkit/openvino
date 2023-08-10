// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/normalize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "normalize_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct normalize_test_params {
    layout input_layout;
    bool across_spatial;
};

class normalize_test : public testing::TestWithParam<normalize_test_params> { };

TEST_P(normalize_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto mem = engine.allocate_memory(layout({1}, p.input_layout.data_type, format::bfyx));
    auto input1_layout_prim = std::make_shared<data>("input1", mem);
    auto normalize_prim = std::make_shared<normalize>("output", input_info("input0"), "input1", p.across_spatial);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& normalize_node = prog.get_or_create(normalize_prim);
    program_wrapper::add_connection(prog, input0_layout_node, normalize_node);
    program_wrapper::add_connection(prog, input1_layout_node, normalize_node);
    auto res = normalize_inst::calc_output_layouts<ov::PartialShape>(normalize_node, *normalize_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, normalize_test,
    testing::ValuesIn(std::vector<normalize_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, false},
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, true},
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, false},
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, true},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, false}
    }));

}  // shape_infer_tests
