// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/lrn.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "lrn_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct lrn_test_params {
    layout input_layout;
    uint32_t size;
    float k;
    float alpha;
    float beta;
    lrn_norm_region norm_region;
};

class lrn_test : public testing::TestWithParam<lrn_test_params> { };

TEST_P(lrn_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto lrn_prim = std::make_shared<lrn>("output", input_info("input0"), p.size, p.k, p.alpha, p.beta, p.norm_region);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& lrn_node = prog.get_or_create(lrn_prim);
    program_wrapper::add_connection(prog, input0_layout_node, lrn_node);
    auto res = lrn_inst::calc_output_layouts<ov::PartialShape>(lrn_node, *lrn_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, lrn_test,
    testing::ValuesIn(std::vector<lrn_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, 5, 0.1f, 0.2f, 0.75f, lrn_norm_region::lrn_norm_region_across_channel},
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, 3, 0.1f, 0.2f, 0.75f, lrn_norm_region::lrn_norm_region_within_channel},
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, 6, 0.1f, 0.2f, 0.75f, lrn_norm_region::lrn_norm_region_across_channel},
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 2, 0.1f, 0.2f, 0.75f, lrn_norm_region::lrn_norm_region_within_channel},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, 5, 0.1f, 0.2f, 0.75, lrn_norm_region::lrn_norm_region_across_channel}
    }));

}  // shape_infer_tests
