// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct eltwise_test_params {
    ov::PartialShape input_size;
    data_types input_type;
    data_types input_type2;
    format input_format;
    data_types default_type;
    format default_format;
    eltwise_mode mode;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class EltwiseFusingTest : public ::BaseFusingTest<eltwise_test_params> {
public:
    void execute(eltwise_test_params& p, bool count_reorder = false) {
        auto input_prim = get_mem(get_input_layout(p));
        auto input_prim2 = get_mem(get_input_layout2(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        auto inputs = network_fused.get_input_ids();
        if (std::find(inputs.begin(), inputs.end(), "input") != inputs.end()) {
            network_fused.set_input_data("input", input_prim);
            network_not_fused.set_input_data("input", input_prim);
        }
        if (std::find(inputs.begin(), inputs.end(), "input2") != inputs.end()) {
            network_fused.set_input_data("input2", input_prim2);
            network_not_fused.set_input_data("input2", input_prim2);
        }

        compare(network_not_fused, network_fused, p, count_reorder);
    }

    layout get_input_layout(eltwise_test_params& p) {
        return layout{ p.input_size, p.input_type, p.input_format };
    }

    layout get_input_layout2(eltwise_test_params& p) {
        return layout{ p.input_size, p.input_type2, p.input_format };
    }

    layout get_per_channel_layout(eltwise_test_params& p) {
        return layout{ { 1, p.input_size[1], 1, 1 }, p.default_type, p.default_format  };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Eltwise cases ---------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_ELTWISE_FP32_1         { 2, 16, 4, 4 }, data_types::f32, data_types::f32, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP32_2         { 2, 16, 4, 4 }, data_types::f32, data_types::f32, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP32_3         { 2, 32, 4, 8 }, data_types::f32, data_types::f32, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_4         { 2, 16, 4, 4 }, data_types::f32, data_types::f32, format::bfwzyx,         data_types::f32,  format::bfwzyx,           eltwise_mode::sum
#define CASE_ELTWISE_FP16_1         { 2, 16, 4, 4 }, data_types::f16, data_types::f16, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP16_2         { 2, 16, 4, 4 }, data_types::f16, data_types::f16, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP16_3         { 2, 32, 4, 8 }, data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_4         { 3, 32, 4, 4 }, data_types::f16, data_types::f16, format::fs_b_yx_fsv32,  data_types::f16,  format::fs_b_yx_fsv32,    eltwise_mode::sum
#define CASE_ELTWISE_I8_1           { 2, 16, 4, 4 }, data_types::i8,  data_types::i8,  format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_2           { 2, 16, 4, 4 }, data_types::i8,  data_types::i8,  format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_3           { 2, 16, 4, 4 }, data_types::i8,  data_types::i8,  format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_1           { 2, 16, 4, 4 }, data_types::u8,  data_types::u8,  format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_2           { 2, 16, 4, 4 }, data_types::u8,  data_types::u8,  format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_3           { 2, 16, 4, 4 }, data_types::u8,  data_types::u8,  format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_1    { 2, 16, 4, 4 }, data_types::f32, data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_2    { 2, 16, 4, 4 }, data_types::f32, data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_3    { 2, 32, 4, 4 }, data_types::f32, data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_1    { 2, 16, 4, 4 }, data_types::f16, data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_2    { 2, 16, 4, 4 }, data_types::f16, data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_3    { 2, 32, 4, 4 }, data_types::f16, data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_1      { 2, 16, 4, 4 }, data_types::i8,  data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_2      { 2, 16, 4, 4 }, data_types::i8,  data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_3      { 2, 32, 4, 4 }, data_types::i8,  data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_1      { 2, 16, 4, 4 }, data_types::i8,  data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_2      { 2, 16, 4, 4 }, data_types::i8,  data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_3      { 2, 32, 4, 4 }, data_types::i8,  data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_1      { 2, 16, 4, 4 }, data_types::u8,  data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_2      { 2, 16, 4, 4 }, data_types::u8,  data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_3      { 2, 32, 4, 4 }, data_types::u8,  data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_1      { 2, 16, 4, 4 }, data_types::u8,  data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_2      { 2, 16, 4, 4 }, data_types::u8,  data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_3      { 2, 32, 4, 4 }, data_types::u8,  data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum

#define CASE_ELTWISE_FP32_5         { 1,  5, 4, 4 }, data_types::f32, data_types::f32, format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_6         { 2, 32, 4, 8 }, data_types::f32, data_types::f32, format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_7         { 1, 8, 16, 1 }, data_types::f32, data_types::f32, format::bfyx,          data_types::f32,  format::bfwzyx,          eltwise_mode::sum
#define CASE_ELTWISE_FP16_5         { 2, 32, 4, 8 }, data_types::f16, data_types::f16, format::b_fs_yx_fsv4,  data_types::f16,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_6         { 1, 32, 4, 8 }, data_types::f16, data_types::f16, format::byxf,          data_types::f16,  format::byxf,            eltwise_mode::sum
#define CASE_ELTWISE_I8_4           { 2, 16, 4, 4 }, data_types::i8,  data_types::i8,  format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_U8_4           { 2, 16, 4, 4 }, data_types::u8,  data_types::u8,  format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum

#define CASE_ELTWISE_FP16_7         { 3, 32, 2, 3, 3, 2, 1, 2 }, data_types::f16, data_types::f16, format::bfvuwzyx,  data_types::f16,  format::bfvuwzyx,    eltwise_mode::sum

class eltwise_quantize : public EltwiseFusingTest {};
TEST_P(eltwise_quantize, u8) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("out", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    if (p.default_type == data_types::f16 && p.default_format == format::b_fs_yx_fsv4) {
        tolerance *= 2.f; // Issue: 94154
    }
    execute(p);
}

TEST_P(eltwise_quantize, i8_per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("out", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    if (p.default_type == data_types::f16 && p.default_format == format::b_fs_yx_fsv4) {
        tolerance *= 11.f; // Issue: 94154
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_quantize, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_3, 3, 4 },
    // fsv4
    eltwise_test_params{ CASE_ELTWISE_FP16_5, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_5, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_6, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_I8_4, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_U8_4, 3, 4 },
}));

class eltwise_const_path : public EltwiseFusingTest {};
TEST_P(eltwise_const_path, not_fuse_to_const_eltwise) {
    auto p = GetParam();
    create_topologies(
        data("const1", get_mem(get_input_layout2(p), -10, 10)),
        data("const2", get_mem(get_input_layout2(p), -10, 10)),
        input_layout("input", get_input_layout2(p)),
        eltwise("eltwise", { input_info("const1"), input_info("const2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("input") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_const_path, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 2, 3 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 2, 3 },
    eltwise_test_params{ CASE_ELTWISE_FP32_5, 2, 3 },
    eltwise_test_params{ CASE_ELTWISE_FP32_6, 2, 3 },
    eltwise_test_params{ CASE_ELTWISE_I8_4, 2, 3 },
    eltwise_test_params{ CASE_ELTWISE_U8_4, 2, 3 },
}));

class eltwise_fp32_fsv16 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv16, add) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::b_fs_yx_fsv16, "eltwise_blocked_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fsv16, add_per_element) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_input_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::b_fs_yx_fsv16, "eltwise_blocked_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fsv16, add_broadcast) {
    auto p = GetParam();
    auto eltwise2_layout = layout{ p.default_type, p.default_format, tensor{ 1, 1, get_input_layout(p).spatial(0), 1 } };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(eltwise2_layout, -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::b_fs_yx_fsv16, "eltwise_blocked_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp32_fsv16, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 5 },
}));

class eltwise_fp32_fsv32 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv32, add) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::fs_b_yx_fsv32, "eltwise_fs_b_yx_fsv32" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fsv32, add_per_element) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_input_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::fs_b_yx_fsv32, "eltwise_fs_b_yx_fsv32" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp32_fsv32, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    // There's no optimized eltwise kernel yet for fsv32 layout that supports fused_ops
    // So only activation is fused via legacy mechanism
    eltwise_test_params{ CASE_ELTWISE_FP16_4, 4, 5 },
}));

class eltwise_fp32_fsv4 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv4, add) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::b_fs_yx_fsv4, "eltwise_blocked_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fsv4, add_per_element) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_input_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::b_fs_yx_fsv4, "eltwise_blocked_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp32_fsv4, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP32_5, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_6, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_4,   3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_4,   3, 5 },
}));

class eltwise_fp32_fused_prims : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fused_prims, scale_activation) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("scale", { input_info("eltwise"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::abs),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fused_prims, eltwise_activation) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("eltwise_data", get_mem(get_input_layout2(p), -10, 10)),
        eltwise("eltwise1", { input_info("input"), input_info("input2") }, p.mode, data_types::f32),
        eltwise("eltwise2", { input_info("eltwise1"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("eltwise2"), activation_func::abs),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

TEST_P(eltwise_fp32_fused_prims, eltwise_activation_with_broadcast) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise1", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("eltwise2", { input_info("eltwise1"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("eltwise2"), activation_func::abs),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp32_fused_prims, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_7, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP32_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP32_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_FP16_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_FP16_3, 3, 5 },
    // fsv4
    eltwise_test_params{ CASE_ELTWISE_FP32_5, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_6, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_I8_4, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_U8_4, 3, 5 },
}));

class eltwise_reorder_eltwise_fp32_fused_prims : public EltwiseFusingTest {};
TEST_P(eltwise_reorder_eltwise_fp32_fused_prims, eltwise_activation) {
    auto p = GetParam();
    create_topologies(
        data("const", get_mem(layout{ {p.input_size[2]}, p.input_type, p.input_format }, -10, 10)),     // 1d const
        data("const2", get_mem(layout{ {1, 1, 1, 1, 1, 1}, p.input_type, p.default_format }, -10, 10)), // 6d const
        input_layout("input", get_input_layout(p)),
        eltwise("eltwise1", { input_info("input"), input_info("const") }, p.mode, p.input_type),
        reorder("reorder6d", input_info("eltwise1"), layout{ {p.input_size[0], p.input_size[1], 1, 1, p.input_size[2], p.input_size[3]}, p.input_type, p.default_format }),
        eltwise("eltwise2", { input_info("reorder6d"), input_info("const2") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("eltwise2"), activation_func::abs),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_reorder_eltwise_fp32_fused_prims, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP32_7, 3, 4 },
}));

class eltwise_fp32_scale : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_scale, 6d) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("scale", { input_info("eltwise"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("out", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp32_scale, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP32_4, 3, 4 },
}));

class eltwise_fp16_byxf : public EltwiseFusingTest {};
TEST_P(eltwise_fp16_byxf, add) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        eltwise("add", { input_info("eltwise"), input_info("add_data") }, eltwise_mode::sum),
        activation("activation", input_info("add"), activation_func::negative),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc eltw_impl = { format::byxf, "generic_eltwise_ref" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "eltwise", eltw_impl } }));

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fp16_byxf, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_6, 3, 5 }
}));

class eltwise_no_pitches_same_dims_quantize : public EltwiseFusingTest {};
TEST_P(eltwise_no_pitches_same_dims_quantize, quantize_f32_output) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, p.input_type),
        reorder("out", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_no_pitches_same_dims_quantize, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 4 },
}));

class eltwise_activation_reorder : public EltwiseFusingTest {};
TEST_P(eltwise_activation_reorder, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {input_info("input"), input_info("input2")}, p.mode, p.default_type),
                      activation("activation", input_info("eltwise"), activation_func::relu, {6.0f, 0.0f}),
                      reorder("out",
                              input_info("activation"),
                              p.default_format,
                              data_types::f32,
                              std::vector<float>(),
                              cldnn::reorder_mean_mode::subtract,
                              cldnn::padding(),
                              true));

    tolerance = default_tolerance(p.input_type);
    execute(p, true);
}

class eltwise_activation : public EltwiseFusingTest {};
TEST_P(eltwise_activation, fp16_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, data_types::f16),
        activation("activation", input_info("eltwise"), activation_func::relu, { 6.0f, 0.0f }),
        reorder("out", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_activation_reorder, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 4, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_3, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_1, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_2, 3, 5 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_3, 4, 5 }
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_activation, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP32_FP16_3, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_1, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_2, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_FP32_3, 3, 4 }
}));



class eltwise_quantize_fs_b_yx_fsv32 : public EltwiseFusingTest {};
TEST_P(eltwise_quantize_fs_b_yx_fsv32, fusing_eltwise_quantize_layout) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, p.input_type),
        reorder("out", input_info("quantize"), format::bfyx, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p);
}

#define CASE_ELTWISE_FP16_FS_B_YX     { 1, 32, 4, 4 }, data_types::f16, data_types::f16, format::fs_b_yx_fsv32,  data_types::f16,  format::fs_b_yx_fsv32,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_BATCH_FS_B  { 8, 32, 4, 4 }, data_types::f16, data_types::f16, format::fs_b_yx_fsv32,  data_types::f16,  format::fs_b_yx_fsv32,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_B_FS_YX     { 1, 32, 4, 4 }, data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_BATCH_B_FS  { 8, 32, 4, 4 }, data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_quantize_fs_b_yx_fsv32, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_FS_B_YX, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_B_FS_YX, 3, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_BATCH_FS_B, 4, 4 },
    eltwise_test_params{ CASE_ELTWISE_FP16_BATCH_B_FS, 3, 4 },
}));

class eltwise_quantize_fs_b_yx_fsv32_exception : public EltwiseFusingTest {};
TEST_P(eltwise_quantize_fs_b_yx_fsv32_exception, fusing_eltwise_quantize_layout_exception) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("input2", get_input_layout2(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        eltwise("eltwise", { input_info("input"), input_info("input2") }, p.mode, p.default_type),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, p.input_type),
        activation("activation", input_info("eltwise"), activation_func::negative),
        eltwise("eltwise_second", { input_info("quantize"), input_info("activation") }, p.mode, p.default_type),
        reorder("out", input_info("eltwise_second"), format::bfyx, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_quantize_fs_b_yx_fsv32_exception, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_FS_B_YX, 6, 6 },
    eltwise_test_params{ CASE_ELTWISE_FP16_B_FS_YX, 6, 6 },
    eltwise_test_params{ CASE_ELTWISE_FP16_BATCH_FS_B, 6, 6 },
    eltwise_test_params{ CASE_ELTWISE_FP16_BATCH_B_FS, 6, 6 },
}));

class eltwise_fusing_reorders : public EltwiseFusingTest {
public:
    layout get_input_layout3(eltwise_test_params& p) {
        return layout{ {1, 1, 1, p.input_size[3]}, p.input_type, p.input_format };
    }
};
TEST_P(eltwise_fusing_reorders, reorders_for_data_type) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("data", get_mem(get_input_layout3(p))),
        eltwise("eltwise", { input_info("input"), input_info("data") }, p.mode, p.default_type),
        reorder("reorder1", input_info("eltwise"), format::bfyx, data_types::i32, {}, reorder_mean_mode::subtract, padding(), true),
        reorder("reorder2", input_info("reorder1"), format::bfyx, data_types::f16, {}, reorder_mean_mode::subtract, padding(), true),
        data("data2", get_mem(get_input_layout3(p))),
        eltwise("eltwise_min", { input_info("reorder2"), input_info("data2") }, eltwise_mode::min, p.default_type),
        reorder("out", input_info("eltwise_min"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.input_type);
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_fusing_reorders, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ { 1, 16, 16, 2 }, data_types::f16, data_types::f16, format::bfyx,  data_types::f16,  format::bfyx, eltwise_mode::max, 4, 6 },
}));

class eltwise_with_constant_input : public EltwiseFusingTest {};
TEST_P(eltwise_with_constant_input, basic) {
    auto p = GetParam();
    create_topologies(data("eltwise_data", get_mem(get_input_layout2(p), -10, 10)),
                      data("eltwise_data1", get_mem(get_input_layout2(p), -10, 10)),
                      eltwise("eltwise", {input_info("eltwise_data"), input_info("eltwise_data1")}, p.mode, p.default_type),
                      reorder("out",
                              input_info("eltwise"),
                              p.default_format,
                              data_types::f32,
                              std::vector<float>(),
                              cldnn::reorder_mean_mode::subtract,
                              cldnn::padding(),
                              true)
                              );

    tolerance = default_tolerance(p.input_type);
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, eltwise_with_constant_input, ::testing::ValuesIn(std::vector<eltwise_test_params>{
    eltwise_test_params{ CASE_ELTWISE_FP16_1, 0, 0},
}));
