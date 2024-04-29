// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/dft.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "dft_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct dft_test_params {
    layout input_layout;
    layout axes_layout;
    layout signal_size_layout;
    std::vector<int64_t> axes_data;
    std::vector<int64_t> signal_size_data;
    bool is_real;
    bool is_inverse;
    bool is_constant;
    layout expected_layout;
};

class dft_test_2_inputs : public testing::TestWithParam<dft_test_params> { };

TEST_P(dft_test_2_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();
    std::vector<cldnn::layout> res;

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    if (p.is_constant) {
        auto axes_mem = engine.allocate_memory(p.axes_layout);
        set_values(axes_mem, p.axes_data);
        auto axes_prim = std::make_shared<data>("axes", axes_mem);

        auto dft_prim = std::make_shared<dft>("dft", input_info("input"), input_info("axes"), p.axes_data,
                                            p.is_inverse ? cldnn::dft_direction::inverse : cldnn::dft_direction::forward,
                                            p.is_real ? cldnn::dft_mode::real : cldnn::dft_mode::complex);

        cldnn::program prog(engine);

        auto& input_node = prog.get_or_create(input_layout_prim);
        auto& axes_node = prog.get_or_create(axes_prim);
        auto& dft_node = prog.get_or_create(dft_prim);
        program_wrapper::add_connection(prog, input_node, dft_node);
        program_wrapper::add_connection(prog, axes_node, dft_node);

        auto params = dft_node.get_kernel_impl_params();

        res = dft_inst::calc_output_layouts<ov::PartialShape>(dft_node, *params);
    } else {
        auto axes_layout_prim = std::make_shared<input_layout>("axes", p.axes_layout);
        std::vector<int64_t> axes;

        auto dft_prim = std::make_shared<dft>("dft", input_info("input"), input_info("axes"), axes,
                                            p.is_inverse ? cldnn::dft_direction::inverse : cldnn::dft_direction::forward,
                                            p.is_real ? cldnn::dft_mode::real : cldnn::dft_mode::complex);

        cldnn::program prog(engine);

        auto axes_mem = engine.allocate_memory(p.axes_layout);
        set_values(axes_mem, p.axes_data);

        auto& input_node = prog.get_or_create(input_layout_prim);
        auto& axes_node = prog.get_or_create(axes_layout_prim);
        auto& dft_node = prog.get_or_create(dft_prim);
        program_wrapper::add_connection(prog, input_node, dft_node);
        program_wrapper::add_connection(prog, axes_node, dft_node);

        auto params = dft_node.get_kernel_impl_params();
        params->memory_deps = {{1, axes_mem}};

        res = dft_inst::calc_output_layouts<ov::PartialShape>(dft_node, *params);
    }

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, dft_test_2_inputs,
    testing::ValuesIn(std::vector<dft_test_params>{
        // DFT
        {
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            false,
            false,
            false,
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            false,
            false,
            true,
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx}
        },
        // IDFT
        {
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            false,
            true,
            false,
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            false,
            true,
            true,
            layout{ov::PartialShape{1, 320, 320, 2}, data_types::f32, format::bfyx}
        },
        // RDFT
        {
            layout{ov::PartialShape{1, 320, 320}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            true,
            false,
            false,
            layout{ov::PartialShape{1, 320, 161, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 320, 320}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            true,
            false,
            true,
            layout{ov::PartialShape{1, 320, 161, 2}, data_types::f32, format::bfyx}
        },
        // IRDFT
        {
            layout{ov::PartialShape{1, 161, 161, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            true,
            true,
            false,
            layout{ov::PartialShape{1, 161, 320}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{1, 161, 161, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
            layout{},
            {1, 2},
            {},
            true,
            true,
            true,
            layout{ov::PartialShape{1, 161, 320}, data_types::f32, format::bfyx}
        },
    }));

class dft_test_3_inputs : public testing::TestWithParam<dft_test_params> { };

TEST_P(dft_test_3_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();
    std::vector<cldnn::layout> res;

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    if (p.is_constant) {
        auto axes_mem = engine.allocate_memory(p.axes_layout);
        auto signal_size_mem = engine.allocate_memory(p.signal_size_layout);
        set_values(axes_mem, p.axes_data);
        set_values(signal_size_mem, p.signal_size_data);
        auto axes_prim = std::make_shared<data>("axes", axes_mem);
        auto signal_size_prim = std::make_shared<data>("signal_size", signal_size_mem);

        auto dft_prim = std::make_shared<dft>("dft", input_info("input"), input_info("axes"), input_info("signal_size"),
                                            p.axes_data, p.signal_size_data,
                                            p.is_inverse ? cldnn::dft_direction::inverse : cldnn::dft_direction::forward,
                                            p.is_real ? cldnn::dft_mode::real : cldnn::dft_mode::complex);

        cldnn::program prog(engine);

        auto& input_node = prog.get_or_create(input_layout_prim);
        auto& axes_node = prog.get_or_create(axes_prim);
        auto& signal_size_node = prog.get_or_create(signal_size_prim);
        auto& dft_node = prog.get_or_create(dft_prim);
        program_wrapper::add_connection(prog, input_node, dft_node);
        program_wrapper::add_connection(prog, axes_node, dft_node);
        program_wrapper::add_connection(prog, signal_size_node, dft_node);

        auto params = dft_node.get_kernel_impl_params();

        res = dft_inst::calc_output_layouts<ov::PartialShape>(dft_node, *params);
    } else {
        auto axes_layout_prim = std::make_shared<input_layout>("axes", p.axes_layout);
        auto signal_size_layout_prim = std::make_shared<input_layout>("signal_size", p.signal_size_layout);
        std::vector<int64_t> axes, signal_size;

        auto dft_prim = std::make_shared<dft>("dft", input_info("input"), input_info("axes"), input_info("signal_size"),
                                            p.axes_data, p.signal_size_data,
                                            p.is_inverse ? cldnn::dft_direction::inverse : cldnn::dft_direction::forward,
                                            p.is_real ? cldnn::dft_mode::real : cldnn::dft_mode::complex);

        cldnn::program prog(engine);

        auto axes_mem = engine.allocate_memory(p.axes_layout);
        auto signal_size_mem = engine.allocate_memory(p.signal_size_layout);
        set_values(axes_mem, p.axes_data);
        set_values(signal_size_mem, p.axes_data);

        auto& input_node = prog.get_or_create(input_layout_prim);
        auto& axes_node = prog.get_or_create(axes_layout_prim);
        auto& signal_size_node = prog.get_or_create(signal_size_layout_prim);
        auto& dft_node = prog.get_or_create(dft_prim);
        program_wrapper::add_connection(prog, input_node, dft_node);
        program_wrapper::add_connection(prog, axes_node, dft_node);
        program_wrapper::add_connection(prog, signal_size_node, dft_node);

        auto params = dft_node.get_kernel_impl_params();
        params->memory_deps = {{1, axes_mem}};
        params->memory_deps = {{2, signal_size_mem}};

        res = dft_inst::calc_output_layouts<ov::PartialShape>(dft_node, *params);
    }

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, dft_test_3_inputs,
    testing::ValuesIn(std::vector<dft_test_params>{
        // DFT
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            false,
            false,
            false,
            layout{ov::PartialShape{16, 768, 2056, 258, 2}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            false,
            false,
            true,
            layout{ov::PartialShape{16, 768, 2056, 258, 2}, data_types::f32, format::bfzyx}
        },
        // IDFT
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            false,
            true,
            false,
            layout{ov::PartialShape{16, 768, 2056, 258, 2}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            false,
            true,
            true,
            layout{ov::PartialShape{16, 768, 2056, 258, 2}, data_types::f32, format::bfzyx}
        },
        // RDFT
        {
            layout{ov::PartialShape{16, 768, 580, 320}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            true,
            false,
            false,
            layout{ov::PartialShape{16, 768, 1029, 258, 2}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape{16, 768, 580, 320}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            true,
            false,
            true,
            layout{ov::PartialShape{16, 768, 1029, 258, 2}, data_types::f32, format::bfzyx}
        },
        // IRDFT
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            true,
            true,
            false,
            layout{ov::PartialShape{16, 768, 2056, 258}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{16, 768, 580, 320, 2}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx},
            {3, 0, 2},
            {258, -1, 2056},
            true,
            true,
            true,
            layout{ov::PartialShape{16, 768, 2056, 258}, data_types::f32, format::bfyx}
        },
    }));
}  // shape_infer_tests
