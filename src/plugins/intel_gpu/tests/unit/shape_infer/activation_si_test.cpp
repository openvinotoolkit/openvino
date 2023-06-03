// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "activation_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct activation_test_params {
    layout input_layout;
    activation_func func;
};

class activation_test : public testing::TestWithParam<activation_test_params> { };

TEST_P(activation_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto activation_prim = std::make_shared<activation>("output", input_info("input"), p.func);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& activation_node = prog.get_or_create(activation_prim);
    program_wrapper::add_connection(prog, input_layout_node, activation_node);
    auto res = activation_inst::calc_output_layouts<ov::PartialShape>(activation_node, *activation_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.input_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, activation_test,
    testing::ValuesIn(std::vector<activation_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, activation_func::relu },
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, activation_func::abs },
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, activation_func::elu },
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, activation_func::erf},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, activation_func::swish}
    }));

}  // shape_infer_tests
