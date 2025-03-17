// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "range_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct range_si_test_params {
    ov::PartialShape input_pshape;
    ov::PartialShape expected_out_pshape;
    data_types out_data_type;
    std::vector<double> vals;   // {start, stop, step}
};

class range_si_test : public testing::TestWithParam<range_si_test_params> { };

TEST_P(range_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    std::vector<layout> input_layouts;
    const size_t num_inputs = 3;

    for (size_t idx = 0; idx < num_inputs; idx++) {
        auto in_layout = layout{p.input_pshape, p.out_data_type, format::bfyx};
        input_layouts.push_back(in_layout);

        auto prim_id = "const::data_" + std::to_string(idx);
        auto const_data_prim = std::make_shared<input_layout>(prim_id, in_layout);
        input_prims.push_back(const_data_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto range_prim = std::make_shared<range>("range", input_prim_ids, layout{p.expected_out_pshape, p.out_data_type, format::bfyx});
    auto& range_node = prog.get_or_create(range_prim);

    for (auto& iprim : input_prims) {
        auto& input_node = prog.get_or_create(iprim);
        program_wrapper::add_connection(prog, input_node, range_node);
    }

    auto params = range_node.get_kernel_impl_params();
    params->memory_deps.clear();
    for (size_t idx = 0; idx < num_inputs; idx++) {
        auto in_layout = input_layouts[idx];
        if (in_layout.is_static() && (idx < p.vals.size())) {
            auto prim_mem = engine.allocate_memory(in_layout);
            ASSERT_NE(p.out_data_type, data_types::dynamic);
            switch (p.out_data_type) {
                case data_types::f16:
                    set_values(prim_mem, {ov::float16(p.vals[idx]).to_bits()});
                    break;
                case data_types::f32:
                    set_values(prim_mem, {static_cast<ov::element_type_traits<data_types::f32>::value_type>(p.vals[idx])});
                    break;
                case data_types::i32:
                    set_values(prim_mem, {static_cast<ov::element_type_traits<data_types::i32>::value_type>(p.vals[idx])});
                    break;
                case data_types::i64:
                    set_values(prim_mem, {static_cast<ov::element_type_traits<data_types::i64>::value_type>(p.vals[idx])});
                    break;
                case data_types::i8:
                    set_values(prim_mem, {static_cast<ov::element_type_traits<data_types::i8>::value_type>(p.vals[idx])});
                    break;
                case data_types::u8:
                    set_values(prim_mem, {static_cast<ov::element_type_traits<data_types::u8>::value_type>(p.vals[idx])});
                    break;
                default:
                    break;
            }
            params->memory_deps.emplace(idx, prim_mem);
        }
    }

    auto res = range_inst::calc_output_layouts<ov::PartialShape>(range_node, *params);

    auto expected_out_layout = layout{p.expected_out_pshape, p.out_data_type, format::bfyx};
    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], expected_out_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, range_si_test,
    testing::ValuesIn(std::vector<range_si_test_params>{
        {ov::PartialShape{}, ov::PartialShape{7}, data_types::i32, {2, 23, 3}},
        {ov::PartialShape{}, ov::PartialShape{7}, data_types::i8,  {2, 23, 3}},
        {ov::PartialShape{}, ov::PartialShape{7}, data_types::u8,  {2, 23, 3}},
        {ov::PartialShape{}, ov::PartialShape{7}, data_types::i64, {23, 2, -3}},
        {ov::PartialShape{}, ov::PartialShape{7}, data_types::i32, {23, 2, -3}},
        {ov::PartialShape{}, ov::PartialShape{3}, data_types::f32, {1.0f, 2.5f, 0.5f}},
        {ov::PartialShape{}, ov::PartialShape{3}, data_types::f16, {1.0f, 2.5f, 0.5f}},
        {ov::PartialShape::dynamic(1), ov::PartialShape::dynamic(1), data_types::f16, {}},
        {ov::PartialShape::dynamic(1), ov::PartialShape::dynamic(1), data_types::i8, {}}
    }));

};  // shape_infer_tests
