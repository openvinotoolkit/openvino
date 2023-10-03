// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "quantize_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct quantize_test_params {
    layout data_layout;
    layout expected_layout;
};

class quantize_test : public testing::TestWithParam<quantize_test_params> { };

TEST_P(quantize_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto in_lo_mem = engine.allocate_memory(layout{ov::PartialShape{1}, p.data_layout.data_type, format::bfyx});
    auto in_hi_mem = engine.allocate_memory(layout{ov::PartialShape{1}, p.data_layout.data_type, format::bfyx});
    auto out_lo_mem = engine.allocate_memory(layout{ov::PartialShape{1}, p.data_layout.data_type, format::bfyx});
    auto out_hi_mem = engine.allocate_memory(layout{ov::PartialShape{1}, p.data_layout.data_type, format::bfyx});

    auto input_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto in_lo_prim = std::make_shared<data>("in_lo", in_lo_mem);
    auto in_hi_prim = std::make_shared<data>("in_hi", in_hi_mem);
    auto out_lo_prim = std::make_shared<data>("out_lo", out_lo_mem);
    auto out_hi_prim = std::make_shared<data>("out_hi", out_hi_mem);
    auto quantize_prim = std::make_shared<quantize>("output", input_info("data"), input_info("in_lo"), input_info("in_hi"),
                                                    input_info("out_lo"), input_info("out_hi"), 255, p.expected_layout.data_type);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& in_lo_node = prog.get_or_create(in_lo_prim);
    auto& in_hi_node = prog.get_or_create(in_hi_prim);
    auto& out_lo_node = prog.get_or_create(out_lo_prim);
    auto& out_hi_node = prog.get_or_create(out_hi_prim);
    auto& quantize_node = prog.get_or_create(quantize_prim);
    program_wrapper::add_connection(prog, input_node, quantize_node);
    program_wrapper::add_connection(prog, in_lo_node, quantize_node);
    program_wrapper::add_connection(prog, in_hi_node, quantize_node);
    program_wrapper::add_connection(prog, out_lo_node, quantize_node);
    program_wrapper::add_connection(prog, out_hi_node, quantize_node);

    auto params = quantize_node.get_kernel_impl_params();
    auto res = quantize_inst::calc_output_layouts<ov::PartialShape>(quantize_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, quantize_test,
    testing::ValuesIn(std::vector<quantize_test_params>{
        {
            layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{1, 2, 3, 4}, data_types::i8, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{1, 2, 3, 4}, data_types::u1, format::b_fs_yx_32fp}
        },
        {
            layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::u8, format::bfzyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::i8, format::bfyx},
        },
        {
            layout{ov::PartialShape{2, 3}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{2, 3}, data_types::i8, format::bfyx}
        }
    }));

}  // shape_infer_tests
