// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/kv_cache.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kv_cache_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct kv_cache_test_params {
    std::vector<layout> input_layouts;
    int64_t concat_axis;
    int64_t gather_axis;
    bool indirect;
    layout expected_layout;
};

class kv_cache_test : public testing::TestWithParam<kv_cache_test_params> { };

TEST_P(kv_cache_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<input_layout>> input_prims;
    std::vector<input_info> input_prims_ids;
    for (size_t i = 0; i < p.input_layouts.size(); i++) {
        auto prim_id = "data" + std::to_string(i);
        auto data_layout_prim = std::make_shared<input_layout>(prim_id, p.input_layouts[i]);
        input_prims.push_back(data_layout_prim);
        input_prims_ids.push_back(input_info(prim_id));
    }

    ov::op::util::VariableInfo info{p.input_layouts[0].get_partial_shape(), p.input_layouts[0].data_type, "v0"};

    auto kv_cache_prim = std::make_shared<kv_cache>("output", input_prims_ids, info, p.concat_axis, p.gather_axis, p.indirect);
    auto& kv_cache_node = prog.get_or_create(kv_cache_prim);
    for (size_t i = 0; i < p.input_layouts.size(); i++) {
        auto& input_node = prog.get_or_create(input_prims[i]);
        program_wrapper::add_connection(prog, input_node, kv_cache_node);
    }

    auto params = kv_cache_node.get_kernel_impl_params();
    auto res = kv_cache_inst::calc_output_layouts<ov::PartialShape>(kv_cache_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, kv_cache_test,
    testing::ValuesIn(std::vector<kv_cache_test_params>{
        {
            {
                layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
                layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx},
            },
            2,
            0,
            false,
            layout{ov::PartialShape{-1, 2, -1, 4}, data_types::f32, format::bfyx}
        },
        {
            {
                layout{ov::PartialShape{1, 2, 0, 4}, data_types::f16, format::bfyx},
                layout{ov::PartialShape{1, 2, 10, 4}, data_types::f16, format::bfyx},
            },
            2,
            0,
            false,
            layout{ov::PartialShape{1, 2, 10, 4}, data_types::f16, format::bfyx}
        },
    }));

}  // shape_infer_tests
