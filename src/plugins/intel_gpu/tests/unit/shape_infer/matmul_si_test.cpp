// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gemm_inst.h"
#include "fully_connected_inst.h"

#include "pass_manager.h"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct matmul_test_params {
    layout matrix_a_layout;
    layout matrix_b_layout;
    data_types data_type;
    bool transpose_a;
    bool transpose_b;
    layout expected_layout;
};

class gemm_test : public testing::TestWithParam<matmul_test_params> {};

TEST_P(gemm_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto matrix_a_layout_prim = std::make_shared<input_layout>("matrix_a", p.matrix_a_layout);
    auto matrix_b_layout_prim = std::make_shared<input_layout>("matrix_b", p.matrix_b_layout);
    auto gemm_prim = std::make_shared<gemm>("output", std::vector<input_info>{ input_info("matrix_a"), input_info("matrix_b") },
                                            p.data_type, p.transpose_a, p.transpose_b);

    cldnn::program prog(engine);

    auto& matrix_a_node = prog.get_or_create(matrix_a_layout_prim);
    auto& matrix_b_node = prog.get_or_create(matrix_b_layout_prim);
    auto& gemm_node = prog.get_or_create(gemm_prim);
    program_wrapper::add_connection(prog, matrix_a_node, gemm_node);
    program_wrapper::add_connection(prog, matrix_b_node, gemm_node);

    auto res = gemm_inst::calc_output_layouts<ov::PartialShape>(gemm_node, *gemm_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gemm_test,
    testing::ValuesIn(std::vector<matmul_test_params>{
        {
            layout{ov::PartialShape{5, 10, 1024}, data_types::i8, format::bfyx},
            layout{ov::PartialShape{1000,  1024}, data_types::i8, format::bfyx},
            data_types::f16, false, true,
            layout{ov::PartialShape{5, 10, 1000}, data_types::f16, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            data_types::f32, false, false,
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

class fully_connected_test : public testing::TestWithParam<matmul_test_params> {};

TEST_P(fully_connected_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto matrix_a_layout_prim = std::make_shared<input_layout>("matrix_a", p.matrix_a_layout);
    auto matrix_b_layout_prim = std::make_shared<input_layout>("matrix_b", p.matrix_b_layout);
    auto fully_connected_prim = std::make_shared<fully_connected>("output", input_info("matrix_a"), "matrix_b", "", p.data_type);

    cldnn::program prog(engine);

    auto& matrix_a_node = prog.get_or_create(matrix_a_layout_prim);
    auto& matrix_b_node = prog.get_or_create(matrix_b_layout_prim);
    auto& fully_connected_node = prog.get_or_create(fully_connected_prim);
    program_wrapper::add_connection(prog, matrix_a_node, fully_connected_node);
    program_wrapper::add_connection(prog, matrix_b_node, fully_connected_node);

    auto res = fully_connected_inst::calc_output_layouts<ov::PartialShape>(fully_connected_node, *fully_connected_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, fully_connected_test,
    testing::ValuesIn(std::vector<matmul_test_params>{
        {
            layout{ov::PartialShape{10, 1024}, data_types::i8, format::bfyx},
            layout{ov::PartialShape{1000, 1024}, data_types::i8, format::bfyx},
            data_types::f16, false, false,
            layout{ov::PartialShape{10, 1000}, data_types::f16, format::bfyx}
        },
        {
            layout{ov::PartialShape{10, 1024}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{1000, 1024}, data_types::f32, format::bfyx},
            data_types::i32, false, false,
            layout{ov::PartialShape{10, 1000}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            data_types::f32, false, false,
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

class fully_connected_test_preferred_output_format : public testing::TestWithParam<matmul_test_params> {};
TEST_P(fully_connected_test_preferred_output_format, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto matrix_a_layout_prim = std::make_shared<input_layout>("matrix_a", p.matrix_a_layout);
    auto matrix_b_layout_prim = std::make_shared<input_layout>("matrix_b", p.matrix_b_layout);
    auto fully_connected_prim = std::make_shared<fully_connected>("output", input_info("matrix_a"), "matrix_b", "", p.data_type);

    cldnn::program prog(engine, {ov::intel_gpu::allow_new_shape_infer(true)});

    auto& matrix_a_node = prog.get_or_create(matrix_a_layout_prim);
    auto& matrix_b_node = prog.get_or_create(matrix_b_layout_prim);
    auto& fully_connected_node = prog.get_or_create(fully_connected_prim);
    program_wrapper::add_connection(prog, matrix_a_node, fully_connected_node);
    program_wrapper::add_connection(prog, matrix_b_node, fully_connected_node);

    fully_connected_node.set_preferred_output_fmt(0, cldnn::format::b_fs_yx_fsv16);

    auto res = fully_connected_inst::calc_output_layouts<ov::PartialShape>(fully_connected_node, *fully_connected_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, fully_connected_test_preferred_output_format,
    testing::ValuesIn(std::vector<matmul_test_params>{
        {
            layout{ov::PartialShape{10, 1024}, data_types::i8, format::bfyx},
            layout{ov::PartialShape{1000, 1024}, data_types::i8, format::bfyx},
            data_types::f16, false, false,
            layout{ov::PartialShape{10, 1000}, data_types::f16, format::b_fs_yx_fsv16}
        }
    }));

class gemm_test_preferred_output_format : public testing::TestWithParam<matmul_test_params> {};
TEST_P(gemm_test_preferred_output_format, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto matrix_a_layout_prim = std::make_shared<input_layout>("matrix_a", p.matrix_a_layout);
    auto matrix_b_layout_prim = std::make_shared<input_layout>("matrix_b", p.matrix_b_layout);
    auto gemm_prim = std::make_shared<gemm>("output", std::vector<input_info>{ input_info("matrix_a"), input_info("matrix_b") },
                                            p.data_type, p.transpose_a, p.transpose_b);

    cldnn::program prog(engine, {ov::intel_gpu::allow_new_shape_infer(true)});

    auto& matrix_a_node = prog.get_or_create(matrix_a_layout_prim);
    auto& matrix_b_node = prog.get_or_create(matrix_b_layout_prim);
    auto& gemm_node = prog.get_or_create(gemm_prim);
    program_wrapper::add_connection(prog, matrix_a_node, gemm_node);
    program_wrapper::add_connection(prog, matrix_b_node, gemm_node);

    gemm_node.set_preferred_output_fmt(0, cldnn::format::b_fs_yx_fsv16);

    auto res = gemm_inst::calc_output_layouts<ov::PartialShape>(gemm_node, *gemm_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gemm_test_preferred_output_format,
    testing::ValuesIn(std::vector<matmul_test_params>{
        {
            layout{ov::PartialShape{5, 10, 1024}, data_types::i8, format::bfyx},
            layout{ov::PartialShape{1000,  1024}, data_types::i8, format::bfyx},
            data_types::f16, false, true,
            layout{ov::PartialShape{5, 10, 1000}, data_types::f16, format::b_fs_yx_fsv16}
        }
    }));

}  // shape_infer_tests
