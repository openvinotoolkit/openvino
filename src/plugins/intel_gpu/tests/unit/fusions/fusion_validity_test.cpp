// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/resample.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct fusing_test_params {
    tensor in_shape;
    tensor out_shape;
    data_types data_type;
    format input_format;
    format output_format;
    resample::InterpolateOp::InterpolateMode type;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_RESAMPLE_FSV16_1_5D { 1, 16, 4, 32, 32 }, { 1, 16, 4, 64, 64 }, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx, resample::InterpolateOp::InterpolateMode::NEAREST, data_types::f16, format::bfzyx
#define CASE_RESAMPLE_FSV16_1 { 1, 16, 64, 64 }, { 1, 16, 128, 128 }, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, resample::InterpolateOp::InterpolateMode::NEAREST, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FSV16_2 { 1, 2,  32, 32 }, { 1, 2,   64,  64 }, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, resample::InterpolateOp::InterpolateMode::NEAREST, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FSV32_1 { 1, 16, 32, 32 }, { 1, 16,  64,  64 }, data_types::i8,  format::b_fs_yx_fsv32, format::bfyx, resample::InterpolateOp::InterpolateMode::NEAREST, data_types::i8, format::bfyx
#define CASE_RESAMPLE_FSV32_2 { 1, 2,  32, 32 }, { 1, 2,   64,  64 }, data_types::i8,  format::b_fs_yx_fsv32, format::bfyx, resample::InterpolateOp::InterpolateMode::NEAREST, data_types::i8, format::bfyx

class PrimitiveFusingTest : public ::BaseFusingTest<fusing_test_params> {
public:

    void execute(fusing_test_params& p) {
        cfg_fused.set_property(ov::intel_gpu::allow_static_input_reorder(true));
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(fusing_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_default_layout(fusing_test_params& p) {
        return layout{ p.data_type, p.default_format, p.out_shape, padding{} };
    }

    layout get_per_channel_layout(fusing_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }
};

} // namespace

// This test is to validate fused operation when a fused post-ops has a planar format input while its data input is a blocked format.
// It is expected to replace LT_ALIGNED_READ with LT_UNALIGNED if fused input is planar while generating FUSED_OPS_LOAD in jitter.
class format_mismatch_fusing : public PrimitiveFusingTest {};

TEST_P(format_mismatch_fusing, single_fused_node) {
    auto p = GetParam();
    create_topologies(
        // Fused eltwise contains format mismatch between data input of resample(input_format) and fused eltwise input(default_format)
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_default_layout(p), -10, 10)),
        resample("resample_opt", input_info("input"), p.out_shape, 1, p.type),
        eltwise("eltwise", { input_info("eltwise_data"), input_info("resample_opt") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.output_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc resample_impl = { p.input_format, "resample_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_opt", resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_resample_impl = { p.input_format, "resample_ref" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_opt", ref_resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_eltwise = { p.input_format, "" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise_data", ref_eltwise } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(validate_fusings_gpu, format_mismatch_fusing, ::testing::ValuesIn(std::vector<fusing_test_params>{
    fusing_test_params{ CASE_RESAMPLE_FSV16_1, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_FSV16_2, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_FSV32_1, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_FSV32_2, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_FSV16_1_5D, 3, 4 },
}));

// This test is to check the replace result of mismatched fusing is valid when multiple nodes are fused.
class format_mismatch_multiple_fusing : public PrimitiveFusingTest {};
TEST_P(format_mismatch_multiple_fusing, multiple_fused_node) {
    auto p = GetParam();
    create_topologies(
        // Multiple fused prims which contains format mismatch (input is input_format, eltwise_data is default_format)
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltwise_data", get_mem(get_default_layout(p), -10, 10)),
        resample("resample_prim", input_info("input"), p.out_shape, p.in_shape.feature[0], p.type),
        eltwise("scale", { input_info("resample_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::abs),
        eltwise("eltwise", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc resample_impl = { p.input_format, "resample_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_prim", resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_resample_impl = { p.input_format, "resample_ref" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_prim", ref_resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_eltwise = { p.input_format, "" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise_data", ref_eltwise } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(validate_fusings_gpu, format_mismatch_multiple_fusing, ::testing::ValuesIn(std::vector<fusing_test_params>{
    fusing_test_params{ CASE_RESAMPLE_FSV16_1, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_FSV16_2, 3, 4 }
}));

#define CASE_RESAMPLE_ONNX_4D_FSV16_1 { 1, 16, 64, 64 }, { 1, 16, 128, 128 }, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, data_types::f16, format::bfyx
#define CASE_RESAMPLE_ONNX_5D_FSV16_1 { 1, 16, 16, 16, 16 }, { 1, 16, 32, 32, 32 }, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, data_types::f16, format::bfzyx

class format_mismatch_onnx_fusing : public PrimitiveFusingTest {};

TEST_P(format_mismatch_onnx_fusing, single_fused_node) {
    auto p = GetParam();
    create_topologies(
        // Fused eltwise contains format mismatch between data input of resample(input_format) and fused eltwise input(default_format)
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_default_layout(p), -10, 10)),
        resample("resample_opt", input_info("input"), p.out_shape, 1, p.type),
        eltwise("eltwise", { input_info("eltwise_data"), input_info("resample_opt") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.output_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc resample_impl = { p.input_format, "resample_onnx" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_opt", resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_resample_impl = { p.input_format, "resample_ref" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_opt", ref_resample_impl } }));
    ov::intel_gpu::ImplementationDesc ref_eltwise = { p.input_format, "" };
    cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise_data", ref_eltwise } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(validate_fusings_onnx_gpu, format_mismatch_onnx_fusing, ::testing::ValuesIn(std::vector<fusing_test_params>{
    fusing_test_params{ CASE_RESAMPLE_ONNX_4D_FSV16_1, 3, 4 },
    fusing_test_params{ CASE_RESAMPLE_ONNX_5D_FSV16_1, 3, 4 }
}));
