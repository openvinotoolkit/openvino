// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/runtime/tensor.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::details;
using namespace ::tests;

namespace {
enum class broadcast_kinds {
    none,
    batch,
    feature
};
struct gemm_test_params {
    std::vector<ov::PartialShape> in_shapes;
    ov::PartialShape out_shape;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
    broadcast_kinds broadcast_kind;
    eltwise_mode eltwise_m;
    std::vector<std::vector<uint16_t>> permute_orders; // input0, input1, output
};

class GemmFusingTest : public ::BaseFusingTest<gemm_test_params> {
public:

    void execute(gemm_test_params& p, bool is_dynamic, bool is_caching_test = false) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));

        auto input0_prim = get_mem(get_input_layout(p, 0));
        auto input1_prim = get_mem(get_input_layout(p, 1));

        if (!p.kernel_name.empty()) {
            ov::intel_gpu::ImplementationDesc gemm_ref_impl = { format::bfyx, "gemm_ref", impl_types::ocl  };
            ov::intel_gpu::ImplementationDesc gemm_target_impl = { format::bfyx, p.kernel_name, impl_types::ocl  };
            cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_prim", gemm_target_impl} }));
            cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_prim", gemm_ref_impl} }));
        }

        network::ptr network_not_fused = get_network(this->engine, this->topology_non_fused, cfg_not_fused, get_test_stream_ptr(), is_caching_test);
        network::ptr network_fused = get_network(this->engine, this->topology_fused, cfg_fused, get_test_stream_ptr(), is_caching_test);
        network_fused->set_input_data("input0", input0_prim);
        network_not_fused->set_input_data("input0", input0_prim);
        network_fused->set_input_data("input1", input1_prim);
        network_not_fused->set_input_data("input1", input1_prim);
        if (p.in_shapes.size() > 2) {
            auto input2_prim = get_mem(get_input_layout(p, 2));
            network_fused->set_input_data("input2", input2_prim);
            network_not_fused->set_input_data("input2", input2_prim);
        }

        compare(*network_not_fused, *network_fused, p);
    }

    layout get_input_layout(gemm_test_params& p, int in_no) {
        if (in_no == 0)
            return layout{ p.in_shapes.at(0), p.data_type_in0, p.input_format };
        else if (in_no == 1)
            return layout{ p.in_shapes.at(1), p.data_type_in1, p.input_format };
        else
            return layout{ p.in_shapes.at(2), p.data_type_in2, p.input_format };
    }

    layout get_per_channel_layout(gemm_test_params& p) {
        return layout{ov::PartialShape{ 1, p.in_shapes[0][1], 1, 1 }, p.default_type, p.default_format };
    }

    layout get_output_layout(gemm_test_params& p) {
        return layout{ p.out_shape, p.default_type, p.input_format };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Gemm cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_GEMM_3IN_FP32_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_2 { { 1, 1, 63, 63 }, { 1, 1, 63, 63 }, { 1, 1, 63, 63 } }, { 1, 1, 63, 63 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_3 { { 1, 1, 128, 128 }, { 1, 1, 128, 128 }, { 1, 1, 128, 128 } }, { 1, 1, 128, 128 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_4 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 }, { 1, 2, 128, 256 } }, { 1, 2, 128, 256 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP16_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_2 { { 1, 1, 31, 31 }, { 1, 1, 31, 31 }, { 1, 1, 31, 31 } }, { 1, 1, 31, 31 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_3 { { 1, 1, 64, 64 }, { 1, 1, 64, 64 }, { 1, 1, 64, 64 } }, { 1, 1, 64, 64 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_4 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 }, { 1, 2, 128, 256 } }, { 1, 2, 128, 256 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_S8S8_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_2 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 }, { 1, 2, 128, 256 } }, { 1, 2, 128, 256 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_3 { { 1, 1, 16, 8 }, { 1, 1, 8, 32 }, { 1, 1, 16, 32 } }, { 1, 1, 16, 32 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_FP32_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_2 { { 1, 1, 63, 63 }, { 1, 1, 63, 63 } }, { 1, 1, 63, 63 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_3 { { 1, 1, 128, 128 }, { 1, 1, 128, 128 } }, { 1, 1, 128, 128 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_4 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 } }, { 1, 2, 128, 256 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP16_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_2 { { 1, 1, 31, 31 }, { 1, 1, 31, 31 } }, { 1, 1, 31, 31 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_3 { { 1, 1, 64, 64 }, { 1, 1, 64, 64 } }, { 1, 1, 64, 64 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_4 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 } }, { 1, 2, 128, 256 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_5 { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_3D_1 { { 16, 8, 64 }, { 16, 64, 8 }, { 16, 1, 8 } }, { 16, 8, 8 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_3D_2 { { 1, 8, 64 }, { 1, 64, 8 }, { 1, 1, 1 } }, { 1, 8, 8 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_5D_1 { { 2, 3, 5, 6, 4 }, { 2, 3, 5, 4, 6} }, { 2, 3, 5, 6, 6 }, data_types::f16, data_types::f16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GEMM_2IN_FP16_6D_1 { { 2, 3, 2, 3, 5, 7 }, { 2, 3, 2, 3, 7, 5 } }, { 2, 3, 2, 3, 5, 5 }, data_types::f16, data_types::f16, data_types::f16, format::bfwzyx, data_types::f16, format::bfwzyx

#define CASE_GEMM_2IN_U8U8_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_2 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 } }, { 1, 2, 128, 256 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_3 { { 1, 1, 16, 32 }, { 1, 1, 32, 16 } }, { 1, 1, 32, 32 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8S8_1 { { 1, 1, 2, 4 }, { 1, 1, 4, 8 } }, { 1, 1, 2, 8 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_S8U8_1 { { 1, 2, 128, 64 }, { 1, 2, 64, 256 } }, { 1, 2, 128, 256 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_ELTWISE_2IN_FP32_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_2 { { 1, 1, 1024, 1024 }, { 1, 1, 1024, 1024 } }, { 1, 1, 1024, 1024 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_S8U8_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_2 { { 1, 1, 1024, 1024 }, { 1, 1, 1024, 1024 } }, { 1, 1, 1024, 1024 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_PERMUTES_FUSION_FP16_1 { { 1, 2, 3, 4 }, { 1, 2, 4, 10 } }, { 1, 2, 3, 10 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_PERMUTES_FUSION_FP16_2 { { 3, 1, 31, 13 }, { 3, 1, 13, 3 } }, { 3, 1, 31, 3 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_PERMUTES_FUSION_FP16_3 { { 17, 11, 2, 18 }, { 17, 11, 18, 4 } }, { 17, 11, 2, 4 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_PERMUTES_FUSION_FP16_4 { { 3, 2, 10, 12 }, { 3, 2, 12, 20 } }, { 3, 2, 10, 20 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_PERMUTES_FUSION_FP16_5 { { 3, 2, 16, 32 }, { 3, 2, 32, 16} }, { 3, 2, 16, 16 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_PERMUTES_FUSION_FP16_6 { { 3, 2, 16, 32 },  { 3, 16, 2, 32} }, { 3, 2, 2, 32 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx

class gemm_3in_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_3in_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        input_layout("input2", get_input_layout(p, 2)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_3in_quantize_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_3IN_FP16_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_3, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_4, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_3, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_4, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_3, 4, 5 },
}));

class gemm_2in_quantize_u8 : public GemmFusingTest {};
TEST_P(gemm_2in_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc gemm_impl = { format::bfyx, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_prim", gemm_impl} }));

    tolerance = default_tolerance(data_types::u8);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_quantize_u8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}));

class gemm_2in_quantize_float_in : public GemmFusingTest {};
TEST_P(gemm_2in_quantize_float_in, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::u8);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_quantize_float_in, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 4 },
}));

class gemm_2in_scale : public GemmFusingTest {};
TEST_P(gemm_2in_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false);
}

TEST_P(gemm_2in_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::f16),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_scale, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}));

class gemm_2in_add : public GemmFusingTest {};
TEST_P(gemm_2in_add, eltwise_postop_static) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_partial_shape();
    if (p.broadcast_kind == broadcast_kinds::batch)
        add_data_size[0] = 1;
    else if (p.broadcast_kind == broadcast_kinds::feature)
        add_data_size[1] = 1;
    add_data_layout.set_partial_shape(add_data_size);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        data("add_data", get_mem(add_data_layout, 0.5f)),   // TODO: Meanless setting, iGPU failed in CASE_GEMM_2IN_FP16_5D_1 with get_mem(add_data_layout, 0, 10)
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false);
}

TEST_P(gemm_2in_add, eltwise_postop_dynamic) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_partial_shape();
    if (p.broadcast_kind == broadcast_kinds::batch)
        add_data_size[0] = 1;
    else if (p.broadcast_kind == broadcast_kinds::feature)
        add_data_size[1] = 1;
    add_data_layout.set_partial_shape(add_data_size);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    in_layout0.set_partial_shape(ov::PartialShape::dynamic(p.in_shapes[0].size()));
    in_layout1.set_partial_shape(ov::PartialShape::dynamic(p.in_shapes[1].size()));

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        data("add_data", get_mem(add_data_layout, 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, true);
}

TEST_P(gemm_2in_add, eltwise_postop_cached) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_partial_shape();
    if (p.broadcast_kind == broadcast_kinds::batch)
        add_data_size[0] = 1;
    else if (p.broadcast_kind == broadcast_kinds::feature)
        add_data_size[1] = 1;
    add_data_layout.set_partial_shape(add_data_size);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        data("add_data", get_mem(add_data_layout, 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false, true);
}

TEST_P(gemm_2in_add, eltwise_postop_scalar) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_partial_shape();
    for (size_t i = 0; i < add_data_size.size(); i++)
        add_data_size[i] = 1;
    add_data_layout.set_partial_shape(add_data_size);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        data("add_data", get_mem(add_data_layout, 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false, true);
}

TEST_P(gemm_2in_add, eltwise_postop_scalar_dynamic) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_partial_shape();
    for (size_t i = 0; i < add_data_size.size(); i++)
        add_data_size[i] = 1;
    add_data_layout.set_partial_shape(add_data_size);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    in_layout0.set_partial_shape(ov::PartialShape::dynamic(p.in_shapes[0].size()));
    in_layout1.set_partial_shape(ov::PartialShape::dynamic(p.in_shapes[1].size()));

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        data("add_data", get_mem(add_data_layout, 0.5f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, true, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_add, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4, "", broadcast_kinds::none, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4, "", broadcast_kinds::none, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::batch, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::batch, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::batch, eltwise_mode::sub },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::feature, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", broadcast_kinds::feature, eltwise_mode::sub },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5D_1, 3, 4, "", broadcast_kinds::batch, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5D_1, 3, 4, "", broadcast_kinds::batch, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5D_1, 3, 4, "", broadcast_kinds::batch, eltwise_mode::sub },
    gemm_test_params{ CASE_GEMM_2IN_FP16_6D_1, 3, 4, "", broadcast_kinds::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_6D_1, 3, 4, "", broadcast_kinds::feature, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_6D_1, 3, 4, "", broadcast_kinds::feature, eltwise_mode::sub },
}));

class gemm_2in_dynamic_add : public gemm_2in_add {};
TEST_P(gemm_2in_dynamic_add, add) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        p.expected_fused_primitives++;
        if (!p.kernel_name.empty()) {
            p.expected_not_fused_primitives++;
        }
    }

    cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto eltwise_layout = get_output_layout(p);
    auto eltwise_shape = ov::PartialShape::dynamic(eltwise_layout.get_partial_shape().size());
    if (p.broadcast_kind == broadcast_kinds::batch)
        eltwise_shape[0] = 1;
    else if (p.broadcast_kind == broadcast_kinds::feature)
        eltwise_shape[1] = 1;
    eltwise_layout.set_partial_shape(eltwise_shape);

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    auto in0_pshape = ov::PartialShape::dynamic(p.in_shapes[0].size());
    in0_pshape[2] = p.in_shapes[0][2];
    auto in1_pshape = ov::PartialShape::dynamic(p.in_shapes[1].size());
    in1_pshape[1] = p.in_shapes[1][1];

    in_layout0.set_partial_shape(in0_pshape);
    in_layout1.set_partial_shape(in1_pshape);

    create_topologies(
        input_layout("input0", in_layout0),
        input_layout("input1", in_layout1),
        input_layout("input2", eltwise_layout),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in0_pshape.size(), in1_pshape.size()),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("input2") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_dynamic_add, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_1, 4, 5, "", broadcast_kinds::batch, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_1, 4, 5, "", broadcast_kinds::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_2, 4, 5, "", broadcast_kinds::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_2, 4, 5, "", broadcast_kinds::batch, eltwise_mode::sum },
    // Reference graph can be fused because force_implementation leads to optimize_data = true;
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_1, 4, 4, "gemm_tiled_opt", broadcast_kinds::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3D_1, 4, 4, "gemm_tiled_opt", broadcast_kinds::batch, eltwise_mode::sum }
}));

class gemm_2in_act_scale_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255.f)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        activation("activation", input_info("gemm_prim"), activation_func::exp),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_U8S8_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_S8U8_1, 3, 6 },
}));

class gemm_2in_act_scale_quantize_eltwise_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_eltwise_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255.f)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        activation("activation", input_info("gemm_prim"), activation_func::exp),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_eltwise_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 7 },
}));

class gemm_2in_act_scale_eltwise : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255.f)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::negative),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    if (p.default_type == data_types::f16 && p.kernel_name == "gemm_tiled_opt") {
        tolerance *= 2.1f; // Issue: 94154
    }
    execute(p, false);
}

TEST_P(gemm_2in_act_scale_eltwise, broadcast_eltwise) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255.f)),
        data("eltwise_data", get_mem(get_single_element_layout(p))),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::negative),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    // Onednn impl gives different results for some reason (looks like missing saturation somewhere)
    ov::intel_gpu::ImplementationDesc gemm_impl = { format::bfyx, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemm_impl } }));

    tolerance = default_tolerance(p.default_type);
    if (p.default_type == data_types::f16 && p.kernel_name == "gemm_tiled_opt") {
        tolerance *= 2.1f; // Issue: 94154
    }
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    gemm_2in_act_scale_eltwise,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 6},
        // Reference graph can be fused because force_implementation leads optimize_data(true) in program::set_options()
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_U8S8_2, 3, 3, "gemm_mmad_int8"},
        // gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_2, 3, 3, "gemm_mmad_int8_slm" },   // tolerance issue
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP16_2, 3, 3, "gemm_tiled_opt"},
    }));
