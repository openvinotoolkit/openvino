// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reduce.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct reduce_test_params {
    cldnn::tensor in_shape;
    cldnn::tensor out_shape;
    cldnn::data_types data_type;
    cldnn::format input_format;
    data_types default_type;
    cldnn::format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    cldnn::reduce_mode reduce_mode;
    std::vector<int64_t> reduce_axes;
    bool keep_dims;
    std::string kernel_name;
};

class ReduceFusingTest : public ::BaseFusingTest<reduce_test_params> {
public:
    void execute(reduce_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    void update_out_shape(reduce_test_params& p) {
        size_t rank = p.input_format.dimension();
        for (auto& axis : p.reduce_axes) {
            if (axis >= static_cast<int64_t>(rank))
                throw std::runtime_error("Unsupported reduce test case");

            switch (axis) {
                case 0:  // batch
                    p.out_shape.batch[0] = 1;
                    break;
                case 1:  // feature
                    p.out_shape.feature[0] = 1;
                    break;
                case 2:
                    p.out_shape.spatial[rank - 3] = 1;
                    break;
                case 3:
                    p.out_shape.spatial[rank - 4] = 1;
                    break;
                case 4:
                    p.out_shape.spatial[rank - 5] = 1;
                    break;
                case 5:
                    p.out_shape.spatial[rank - 6] = 1;
                    break;
            }
        }
    }

    layout get_input_layout(reduce_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape };
    }

    layout get_per_channel_layout(reduce_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Reduce cases ----------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_REDUCE_F32_0 { 3, 7, 5, 7 }, { 3, 7, 5, 7 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_1 { 3, 7, 5, 7 }, { 3, 7, 5, 7 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_2 { 2, 4, 8, 4, 4 }, { 2, 4, 8, 4, 4 }, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_3 { 16, 16, 16, 8, 8, 8 }, { 16, 16, 16, 8, 8, 8 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_4 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_F16_0 { 3, 7, 5, 7 }, { 3, 7, 5, 7 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_1 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_2 { 2, 4, 8, 4, 4 }, { 2, 4, 8, 4, 4 }, data_types::f16, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_3 { 3, 5, 3, 5, 7, 7 }, { 3, 5, 3, 5, 7, 7 }, data_types::f16, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_4 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_I32_0 { 3, 7, 5, 7 }, { 3, 7, 5, 7 }, data_types::i32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_1 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::i32, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_2 { 2, 4, 8, 4, 4 }, { 2, 4, 8, 4, 4 }, data_types::i32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_3 { 3, 5, 3, 5, 7, 7 }, { 3, 5, 3, 5, 7, 7 }, data_types::i32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_4 { 3, 5, 3, 5, 7, 7 }, { 3, 5, 3, 5, 7, 7 }, data_types::i32, format::bfwzyx, data_types::f32, format::bfyx

#define CASE_REDUCE_I8_0 { 3, 7, 5, 7 }, { 3, 7, 5, 7 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_1 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_2 { 2, 4, 8, 4, 4 }, { 2, 4, 8, 4, 4 }, data_types::i8, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_3 { 3, 5, 3, 5, 7, 7 }, { 3, 5, 3, 5, 7, 7 }, data_types::i8, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_4 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_U8_0 { 3, 7, 5, 7 }, { 3, 7, 5, 7 },data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_1 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_2 { 2, 4, 8, 4, 4 }, { 2, 4, 8, 4, 4 }, data_types::u8, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_3 { 3, 5, 3, 5, 7, 7 }, { 3, 5, 3, 5, 7, 7 }, data_types::u8, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_4 { 2, 8, 4, 4 }, { 2, 8, 4, 4 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

class reduce_eltwise_activation_quantize : public ReduceFusingTest {};
TEST_P(reduce_eltwise_activation_quantize, basic) {
    auto p = GetParam();
    update_out_shape(p);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        reduce("reduce", input_info("input"), p.reduce_mode, p.reduce_axes, p.keep_dims),
        eltwise("eltwise", { input_info("reduce"), input_info("eltwise_data") }, eltwise_mode::sum, p.default_type),
        activation("activation", input_info("eltwise"), activation_func::relu),
        quantize("quantize", input_info("activation"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("output_reorder", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(reduce_eltwise_activation_quantize, per_channel) {
    auto p = GetParam();
    update_out_shape(p);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        reduce("reduce", input_info("input"), p.reduce_mode, p.reduce_axes, p.keep_dims),
        eltwise("eltwise", { input_info("reduce"), input_info("eltwise_data") }, eltwise_mode::sum, p.default_type),
        activation("activation", input_info("eltwise"), activation_func::relu),
        quantize("quantize", input_info("activation"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("output_reorder", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, reduce_eltwise_activation_quantize, ::testing::ValuesIn(std::vector<reduce_test_params>{
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 5, reduce_mode::mean, { 3, 1, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, { 3, 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 5, reduce_mode::max, { 2, 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, { 3, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 5, reduce_mode::min, { 3, 2, 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_1, 2, 5, reduce_mode::sum, { 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_2, 2, 5, reduce_mode::mean, { 1, 4 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_1, 2, 5, reduce_mode::max, { 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_2, 2, 5, reduce_mode::sum, { 4, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 5, reduce_mode::max, { 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, { 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 5, reduce_mode::min, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_1, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_2, 2, 5, reduce_mode::max, { 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_1, 2, 5, reduce_mode::mean, { 3 }, true, "reduce_ref" },

    reduce_test_params{ CASE_REDUCE_F16_1, 2, 5, reduce_mode::mean, { 3, 1, 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_2, 2, 5, reduce_mode::sum, { 4, 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_1, 2, 5, reduce_mode::max, { 2, 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_2, 2, 5, reduce_mode::sum, { 4, 3, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_1, 2, 5, reduce_mode::min, { 3, 2, 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 5, reduce_mode::sum, { 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_4, 2, 5, reduce_mode::mean, { 1, 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 5, reduce_mode::max, { 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_4, 2, 5, reduce_mode::sum, { 3, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_1, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_2, 2, 5, reduce_mode::max, { 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_1, 2, 5, reduce_mode::sum, { 2 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_2, 2, 5, reduce_mode::min, { 4 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_4, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 5, reduce_mode::max, { 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_4, 2, 5, reduce_mode::mean, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },

    reduce_test_params{ CASE_REDUCE_I8_0, 2, 5, reduce_mode::mean, { 3, 1, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, { 3, 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_0, 2, 5, reduce_mode::max, { 2, 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, { 3, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_0, 2, 5, reduce_mode::min, { 3, 2, 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_1, 2, 5, reduce_mode::sum, { 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_2, 2, 5, reduce_mode::mean, { 1, 3 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_1, 2, 5, reduce_mode::max, { 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_2, 2, 5, reduce_mode::sum, { 3, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_0, 2, 5, reduce_mode::max, { 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, { 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_0, 2, 5, reduce_mode::min, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_I8_1, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_2, 2, 5, reduce_mode::max, { 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_1, 2, 5, reduce_mode::mean, { 3 }, true, "reduce_ref" },

    reduce_test_params{ CASE_REDUCE_U8_1, 2, 5, reduce_mode::mean, { 3, 1, 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_2, 2, 5, reduce_mode::sum, { 4, 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_1, 2, 5, reduce_mode::max, { 2, 1, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_2, 2, 5, reduce_mode::sum, { 4, 3, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_1, 2, 5, reduce_mode::min, { 3, 2, 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_0, 2, 5, reduce_mode::sum, { 1, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_4, 2, 5, reduce_mode::mean, { 1, 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_0, 2, 5, reduce_mode::max, { 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_4, 2, 5, reduce_mode::sum, { 3, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_1, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_2, 2, 5, reduce_mode::max, { 1 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_1, 2, 5, reduce_mode::sum, { 2 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_2, 2, 5, reduce_mode::min, { 4 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_4, 2, 5, reduce_mode::sum, { 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_0, 2, 5, reduce_mode::max, { 1 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_U8_4, 2, 5, reduce_mode::mean, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" }
}));

class reduce_scale_activation : public ReduceFusingTest {};
TEST_P(reduce_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_single_element_layout(p), -0.125f)),
        reduce("reduce", input_info("input"), p.reduce_mode, p.reduce_axes, p.keep_dims),
        eltwise("scale", { input_info("reduce"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation", input_info("scale"), activation_func::cos),
        reorder("output_reorder", input_info("activation"), p.default_format, data_types::f32)
    );
    // Activation won't be fused because onednn doesn't support cos activation
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives++;

    tolerance = 1e-02f;
    execute(p);
}

TEST_P(reduce_scale_activation, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -0.125f)),
        reduce("reduce", input_info("input"), p.reduce_mode, p.reduce_axes, p.keep_dims),
        eltwise("scale", { input_info("reduce"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation", input_info("scale"), activation_func::cos),
        reorder("output_reorder", input_info("activation"), p.default_format, data_types::f32)
    );
    // Activation won't be fused because onednn doesn't support cos activation
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives++;

    tolerance = 1e-02f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, reduce_scale_activation, ::testing::ValuesIn(std::vector<reduce_test_params>{
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::max, { 3, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_1, 2, 4, reduce_mode::sum, { 3, 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::min, { 3, 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_2, 2, 4, reduce_mode::mean, { 4, 3 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::l1, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::l1, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::min, { 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F32_0, 2, 4, reduce_mode::sum, { 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },

    reduce_test_params{ CASE_REDUCE_F16_0, 2, 4, reduce_mode::max, { 3, 2, 0 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_1, 2, 4, reduce_mode::sum, { 3, 2, 0 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 4, reduce_mode::min, { 3, 2 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_2, 2, 4, reduce_mode::mean, { 4, 3 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 4, reduce_mode::min, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
    reduce_test_params{ CASE_REDUCE_F16_0, 2, 4, reduce_mode::sum, { 3 }, true, "reduce_gpu_b_fs_yx_fsv16" },
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, reduce_eltwise_activation_quantize, ::testing::ValuesIn(std::vector<reduce_test_params>{
    // No layout format available for quantize/scale
    reduce_test_params{ CASE_REDUCE_F32_3, 2, 4, reduce_mode::l1, { 5 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_F16_3, 2, 4, reduce_mode::min, { 5 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I32_2, 2, 4, reduce_mode::max, { 4, 3 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I32_3, 2, 4, reduce_mode::sum, { 5 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_I8_3, 2, 4, reduce_mode::mean, { 5 }, true, "reduce_ref" },
    reduce_test_params{ CASE_REDUCE_U8_3, 2, 4, reduce_mode::l2, { 5 }, true, "reduce_ref" }
}));
