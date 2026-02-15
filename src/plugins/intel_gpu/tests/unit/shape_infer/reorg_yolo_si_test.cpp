// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorg_yolo.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "reorg_yolo_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct reorg_yolo_test_params {
    layout in_layout;
    uint32_t stride;
    layout expected_layout;
};

class reorg_yolo_test : public testing::TestWithParam<reorg_yolo_test_params> { };

TEST_P(reorg_yolo_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto reorg_yolo_prim = std::make_shared<reorg_yolo>("reorg_yolo",
                                                         input_info("input"),
                                                         p.stride);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& reorg_yolo_node = prog.get_or_create(reorg_yolo_prim);
    program_wrapper::add_connection(prog, input_layout_node, reorg_yolo_node);
    auto res = reorg_yolo_inst::calc_output_layouts<ov::PartialShape>(reorg_yolo_node, *reorg_yolo_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, reorg_yolo_test,
    testing::ValuesIn(std::vector<reorg_yolo_test_params>{
        {
            layout{ov::PartialShape{1, 64, 26, 26}, data_types::f32, format::bfyx},
            2,
            layout{ov::PartialShape{1, 256, 13, 13}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            2,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
