// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "arg_max_min_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct arg_max_min_test_params {
    std::vector<layout> in_layouts;
    ov::op::TopKMode mode;
    uint32_t top_k;
    int64_t axis;
    data_types output_data_type;
    std::vector<input_info> inputs;
    size_t num_outputs;
    std::vector<layout> expected_layouts;
};

class arg_max_min_test : public testing::TestWithParam<arg_max_min_test_params> { };

TEST_P(arg_max_min_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    {
        auto prim_id = "input";
        auto input_layout_prim = std::make_shared<input_layout>(prim_id, p.in_layouts[0]);
        input_prims.push_back(input_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    for (size_t i = 1; i < p.in_layouts.size(); i++) {
        auto prim_id = "const" + std::to_string(i);
        auto prim_mem = engine.allocate_memory(p.in_layouts[i]);
        if (i == 2 && p.inputs.empty()) {
            auto mutable_data_prim = std::make_shared<mutable_data>(prim_id, prim_mem);
            input_prims.push_back(mutable_data_prim);
        } else {
            auto const_data_prim = std::make_shared<data>(prim_id, prim_mem);
            input_prims.push_back(const_data_prim);
        }
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto arg_max_min_prim = std::make_shared<arg_max_min>("output", p.inputs.empty() ? input_prim_ids : p.inputs,
                                                          p.mode, p.top_k, p.axis,
                                                          ov::op::TopKSortType::SORT_VALUES, false, false,
                                                          p.output_data_type, p.num_outputs);
    std::vector<padding> output_paddings;
    std::vector<optional_data_type> output_data_types;
    for (size_t i = 0; i < p.num_outputs; i++) {
        output_paddings.push_back(padding());
        output_data_types.push_back(optional_data_type{p.output_data_type});
    }
    arg_max_min_prim->output_paddings = output_paddings;
    arg_max_min_prim->output_data_types = output_data_types;
    auto& arg_max_min_node = prog.get_or_create(arg_max_min_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, arg_max_min_node);
    }

    auto res = arg_max_min_inst::calc_output_layouts<ov::PartialShape>(arg_max_min_node, *arg_max_min_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), p.num_outputs);
    for (size_t i = 0; i < p.expected_layouts.size(); i++)
        ASSERT_EQ(res[i], p.expected_layouts[i]);
}

INSTANTIATE_TEST_SUITE_P(smoke, arg_max_min_test,
    testing::ValuesIn(std::vector<arg_max_min_test_params>{
        {
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}},
            ov::op::TopKMode::MIN, 2, 0, data_types::f32, {}, 1,
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}},
            ov::op::TopKMode::MIN, 1, 2, data_types::i32, {}, 1,
            {layout{ov::PartialShape{2, 4, 1, 2}, data_types::i32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{2, 4, 2, 2, 1}, data_types::f32, format::bfzyx}},
            ov::op::TopKMode::MIN, 2, 0, data_types::i32, {}, 1,
            {layout{ov::PartialShape{2, 4, 2, 2, 1}, data_types::i32, format::bfzyx}}
        },
        {
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}},
            ov::op::TopKMode::MIN, 2, 0, data_types::f32, {}, 1,
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx}},
            ov::op::TopKMode::MIN, 2, 0, data_types::f32, {input_info("input", 0), input_info("const0", 0)}, 2,
            {layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{2, 4, 2, 2}, data_types::f32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::f32, format::bfyx}},
            ov::op::TopKMode::MIN, 2, 0, data_types::f32, {input_info("input", 0), input_info("const0", 0)}, 2,
            {layout{ov::PartialShape{2, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                    data_types::f32, format::bfyx},
             layout{ov::PartialShape{2, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                    data_types::f32, format::bfyx}}
        },
    }));

}  // namespace shape_infer_tests
