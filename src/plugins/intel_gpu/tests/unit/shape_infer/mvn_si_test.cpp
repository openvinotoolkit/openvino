// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "mvn_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct mvn_test_params {
    layout input_layout;
    bool normalize_variance;
    float epsilon;
    bool eps_inside_sqrt;
    std::vector<int64_t> axes;
};

class mvn_test : public testing::TestWithParam<mvn_test_params> { };

TEST_P(mvn_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto mvn_prim = std::make_shared<mvn>("output", input_info("input"), p.normalize_variance, p.epsilon, p.eps_inside_sqrt, p.axes);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& mvn_node = prog.get_or_create(mvn_prim);
    program_wrapper::add_connection(prog, input_layout_node, mvn_node);
    auto res = mvn_inst::calc_output_layouts<ov::PartialShape>(mvn_node, *mvn_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.input_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, mvn_test,
    testing::ValuesIn(std::vector<mvn_test_params>{
        {
            layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx},
            true, 1e-9f, true, {2, 3}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            true, 1e-9f, true, {1, 2, 3}
        }
    }));

}  // shape_infer_tests
