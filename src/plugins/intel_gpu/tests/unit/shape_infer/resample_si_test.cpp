// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "resample_inst.h"

#include "program_wrapper.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct resample_test_params {
    layout input;
    std::vector<int64_t> sizes;
    std::vector<float> scales;
    std::vector<int64_t> axes;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    resample::InterpolateOp::InterpolateMode operation_type;
    resample::InterpolateOp::ShapeCalcMode shape_calc_mode;
    resample::InterpolateOp::CoordinateTransformMode ctm;
    resample::InterpolateOp::NearestMode nm;
    layout expected_layout;
};

class resample_test : public testing::TestWithParam<resample_test_params> { };

TEST_P(resample_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.input);
    auto resample_prim = std::make_shared<resample>("output", input_info("input"), p.sizes, p.scales, p.axes, p.pads_begin, p.pads_end, 0, 0,
                                                    p.operation_type, p.shape_calc_mode, p.ctm, p.nm);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& resample_node = prog.get_or_create(resample_prim);
    program_wrapper::add_connection(prog, input_node, resample_node);
    auto res = resample_inst::calc_output_layouts<ov::PartialShape>(resample_node, *resample_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

TEST_P(resample_test, shape_infer_with_fused_op) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.input);
    auto resample_prim = std::make_shared<resample>("output", input_info("input"), p.sizes, p.scales, p.axes, p.pads_begin, p.pads_end, 0, 0,
                                                    p.operation_type, p.shape_calc_mode, p.ctm, p.nm);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& resample_node = prog.get_or_create(resample_prim);
    program_wrapper::add_connection(prog, input_node, resample_node);


    auto expected_layout = p.expected_layout;
    expected_layout.data_type = data_types::u8;

    auto dummy_prim = std::make_shared<activation>("output1", input_info("output"), activation_func::abs);
    fused_primitive_desc desc(dummy_prim);
    desc.output_layout = expected_layout;

    resample_node.add_fused_primitive(desc);

    auto res = resample_inst::calc_output_layouts<ov::PartialShape>(resample_node, *resample_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, resample_test,
    testing::ValuesIn(std::vector<resample_test_params>{
        {
            layout{ov::PartialShape{1, 40, 128, 128}, data_types::f32, format::bfyx},
            std::vector<int64_t>{64, 64},
            std::vector<float>{1.0f, 1.0f},
            std::vector<int64_t>{2, 3},
            std::vector<size_t>{},
            std::vector<size_t>{},
            resample::InterpolateOp::InterpolateMode::NEAREST,
            resample::InterpolateOp::ShapeCalcMode::SIZES,
            resample::InterpolateOp::CoordinateTransformMode::ASYMMETRIC,
            resample::InterpolateOp::NearestMode::SIMPLE,
            layout{ov::PartialShape{1, 40, 64, 64}, data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
