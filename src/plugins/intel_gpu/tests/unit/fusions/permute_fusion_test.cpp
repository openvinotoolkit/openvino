// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/permute.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct permute_params {
    tensor in_shape;
    tensor out_shape;
    std::vector<uint16_t> permute_order;
    tensor eltw_in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

struct permute_reorder_params {
    tensor in_shape;
    std::vector<uint16_t> permute_order1;
    std::vector<uint16_t> permute_order2;
    data_types permute_type;
    data_types output_type;
    format permute_format;
    format output_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class PermuteFusingTest : public ::BaseFusingTest<permute_params> {
public:

    void execute(permute_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(permute_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_per_channel_layout(permute_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }
};

class PermuteReorderFusingTest : public ::BaseFusingTest<permute_reorder_params> {
public:

    void execute(permute_reorder_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(permute_reorder_params& p) {
        return layout{ p.permute_type, p.permute_format, p.in_shape, padding{} };
    }

    layout get_dynamic_input_layout(permute_reorder_params& p) {
        ov::PartialShape pshape = {};
        for (size_t i = 0; i < p.permute_format.dimension(); i++) {
            pshape.push_back(ov::Dimension::dynamic());
        }
        return layout{ pshape, p.permute_type, p.permute_format, padding{} };
    }

    layout get_elt_input_layout(permute_reorder_params&p) {
        ov::Shape output_shape;
        auto input_shape = get_input_layout(p).get_dims();
        for (int32_t o = 0; o < static_cast<int32_t>(p.permute_order1.size()); ++o) {
            output_shape.push_back(input_shape[p.permute_order1[o]]);
        }
        return layout{ ov::PartialShape(output_shape), p.permute_type, p.permute_format, padding{} };
    }

};
}  // namespace

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------------------- PERMUTE FUSE cases ------------------------------------------------ */
/* ------------------------------------------------------------------------------------------------------------ */
#define CASE_PERMUTE_F32_0 { 1, 16, 2, 2 }, { 1, 16, 2, 2 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_1 { 1, 15, 16, 16 }, { 1, 15, 16, 16 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_2 { 1, 8, 16, 16 }, { 16, 16, 8, 1 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_3 { 1, 1, 3, 4 }, { 1, 3, 4, 1 }, { 1, 3, 0, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_4 { 2, 16, 16, 16 }, { 2, 16, 16, 16 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_5 { 1, 32, 4, 5 }, { 32, 4, 5, 1 }, { 1, 3, 0, 2 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_6 { 1, 16, 4, 5 }, { 5, 16, 4, 1 }, { 2, 1, 0, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_7 { 1, 16, 1, 1 }, { 1, 1, 1, 16 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_8 { 1, 16, 1, 32 }, { 1, 16, 1, 32 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_PERMUTE_F16_0 { 1, 16, 4, 5 }, { 1, 16, 4, 5 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_1 { 2, 16, 4, 5 }, { 16, 4, 5, 2 }, { 1, 3, 0, 2 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_2 { 1, 32, 2, 3 }, { 2, 3, 32, 1 }, { 3, 2, 0, 1 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_3 { 3, 16, 1, 1 }, { 1, 1, 16, 3 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_4 { 2, 15, 4, 5 }, { 4, 2, 5, 15 }, { 3, 0, 1, 2 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_5 { 1, 15, 1, 2 }, { 15, 2, 1, 1 }, { 1, 2, 0, 3 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_6 { 1, 15, 4, 4 }, { 4, 4, 1, 15 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx

#define CASE_PERMUTE_S8_0 { 1, 15, 4, 5 }, { 1, 15, 4, 5 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_1 { 1, 15, 4, 5 }, { 5, 4, 15, 1 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_2 { 1, 16, 1, 2 }, { 1, 1, 16, 2 }, { 3, 0, 2, 1 }, tensor{ 0 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_3 { 1, 16, 2, 2 }, { 2, 2, 16, 1 }, { 3, 2, 0, 1 }, tensor{ 0 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_0 { 1, 15, 4, 5 }, { 15, 5, 1, 4 }, { 1, 2, 3, 0 }, tensor{ 0 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_1 { 1, 15, 16, 16 }, { 15, 16, 1, 16 }, { 1, 3, 2, 0 }, tensor{ 0 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_2 { 1, 32, 5, 4 }, { 1, 32, 5, 4 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_3 { 1, 16, 4, 5 }, { 5, 4, 16, 1 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

// 3d
#define CASE_PERMUTE_F32_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_1 { 2, 15, 2, 3, 4 }, { 15, 2, 3, 4, 2 }, { 1, 4, 0, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_2 { 2, 16, 4, 4, 5 }, { 4, 2, 4, 5, 16 }, { 3, 0, 1, 2, 4 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_3 { 1, 32, 4, 2, 2 }, { 2, 2, 32, 1, 4 }, { 2, 3, 4, 0, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_4 { 1, 16, 1, 1, 1 }, { 1, 1, 1, 16, 1 }, { 4, 2, 3, 1, 0 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_F16_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_1 { 2, 15, 4, 3, 4 }, { 4, 4, 2, 15, 3 }, { 4, 2, 3, 1, 0 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_2 { 2, 16, 4, 4, 3 }, { 2, 4, 3, 16, 4 }, { 0, 3, 4, 1, 2 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_3 { 1, 32, 4, 2, 1 }, { 2, 32, 4, 1, 1 }, { 3, 1, 0, 2, 4 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_4 { 16, 16, 1, 1, 1 },{ 1, 16, 1, 1, 16 },{ 2, 0, 1, 4, 3 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_S8_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_1 { 2, 15, 4, 3, 4 }, { 4, 4, 15, 2, 3 }, { 2, 4, 3, 0, 1 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_2 { 2, 16, 4, 4, 3 }, { 2, 4, 3, 16, 4 }, { 0, 3, 4, 1, 2 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_3 { 1, 32, 4, 2, 1 }, { 2, 32, 4, 1, 1 }, { 3, 1, 0, 2, 4 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_0 { 16, 16, 1, 1, 1 }, { 1, 1, 16, 16, 1 }, { 4, 2, 3, 1, 0 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_1 { 16, 16, 1, 1, 1 }, { 1, 1, 1, 16, 16 }, { 2, 3, 0, 1, 4 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_2 { 2, 16, 4, 4, 3 }, { 4, 2, 4, 3, 16 }, { 3, 0, 1, 2, 4 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_3 { 1, 32, 4, 2, 1 }, { 1, 2, 32, 1, 4 }, { 2, 3, 4, 0, 1 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx

// permute_tile_8x8_4x4
#define CASE_PERMUTE_TILE_8x8_4x4_4D_0 { 1, 8, 8, 2 }, { 1, 2, 8, 8 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_1 { 1, 5, 8, 2 }, { 1, 2, 5, 8 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_2 { 1, 8, 5, 2 }, { 1, 2, 8, 5 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_3 { 1, 5, 5, 2 }, { 1, 2, 5, 5 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_0 { 1, 8, 8, 2, 2 }, { 1, 2, 8, 8, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_1 { 1, 5, 8, 2, 2 }, { 1, 2, 5, 8, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_2 { 1, 8, 5, 2, 2 }, { 1, 2, 8, 5, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_3 { 1, 5, 5, 2, 2 }, { 1, 2, 5, 5, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_0 { 1, 8, 8, 2, 2, 2 }, { 1, 2, 8, 8, 2, 2 }, { 0, 2, 3, 4, 5, 1 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_1 { 1, 5, 8, 2, 2, 2 }, { 1, 2, 5, 8, 2, 2 }, { 0, 2, 3, 4, 5, 1 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_2 { 1, 8, 5, 2, 2, 2 }, { 1, 2, 8, 5, 2, 2 }, { 0, 2, 3, 4, 5, 1 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_3 { 1, 5, 5, 2, 2, 2 }, { 1, 2, 5, 5, 2, 2 }, { 0, 2, 3, 4, 5, 1 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx

// permute_tile_8x8_4x4_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0 { 1, 16, 16, 2 }, { 1, 2, 16, 16 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1 { 1, 15, 16, 2 }, { 1, 2, 15, 16 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2 { 1, 16,  3, 2 }, { 1, 2, 16,  3 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3 { 1,  5,  7, 2 }, { 1, 2,  5,  7 }, { 0, 2, 3, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0 { 1, 16, 16, 2, 2 }, { 1, 2, 16, 16, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1 { 1, 15, 16, 2, 2 }, { 1, 2, 15, 16, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2 { 1, 16,  3, 2, 2 }, { 1, 2, 16,  3, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3 { 1,  5,  7, 2, 2 }, { 1, 2,  5,  7, 2 }, { 0, 2, 3, 4, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16

// permute_bfzyx_to_bfyxz
#define CASE_PERMUTE_TILE_BFZYX_TO_BFYXZ_0 { 1, 8, 8, 2, 2 }, { 1, 8, 2, 8, 2 }, { 0, 1, 3, 4, 2 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

// permute_f_y_axes
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_0 { 1, 8, 4, 2 }, { 1, 2, 4, 8 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_1 { 1, 32, 256, 512 }, { 1, 512, 256, 32 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv32, data_types::f32, format::b_fs_yx_fsv32
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_2 {1, 4, 1, 8 }, { 1, 8, 1, 4 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::b_fs_yx_fsv8
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_3 {1, 64, 1, 32 }, { 1, 32, 1, 64 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv32, data_types::f32, format::b_fs_yx_fsv32
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_4 {1, 2, 1, 4 }, { 1, 4, 1, 2 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_5 {1, 4, 1, 8 }, { 1, 8, 1, 4 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_6 {1, 2, 1, 8 }, { 1, 8, 1, 2 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_7 {1, 16, 1, 8 }, { 1, 8, 1, 16 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_8 {1, 4, 1, 32 }, { 1, 32, 1, 4 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::b_fs_yx_fsv32
#define CASE_PERMUTE_TILE_BFYX_TO_BYFX_9 {1, 16, 1, 2 }, { 1, 2, 1, 16 }, { 0, 2, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx

// permute_xy_swap
// Order {0,1,3,2} on bfyx with X and Y both divisible by 16 (or 32) selects PermuteKernel_xy_swap
// (FORCE_PRIORITY_2). Tensor argument order is {B, F, X, Y}; order {0,1,3,2} swaps X<->Y, so out
// shape is {B, F, Y, X}.
#define CASE_PERMUTE_XY_SWAP_F32_0 { 1, 8, 16, 16 }, { 1, 8, 16, 16 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_F32_1 { 1, 4, 32, 32 }, { 1, 4, 32, 32 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_F32_2 { 2, 4, 32, 16 }, { 2, 4, 16, 32 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_F16_0 { 1, 8, 16, 16 }, { 1, 8, 16, 16 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_F16_1 { 1, 4, 32, 32 }, { 1, 4, 32, 32 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_S8_0 { 1, 8, 16, 16 }, { 1, 8, 16, 16 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_U8_0 { 1, 8, 16, 16 }, { 1, 8, 16, 16 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx

class permute_activation_scale_eltwise: public PermuteFusingTest {};
TEST_P(permute_activation_scale_eltwise, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape })),
        data("scale_data", get_mem(get_per_channel_layout(p), 5e-1f)),
        permute("permute", input_info("input"), p.permute_order),
        eltwise("scale", { input_info("permute"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("actv", input_info("scale"), activation_func::relu),
        eltwise("eltwise", { input_info("actv"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_activation_scale_eltwise, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 5 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 5 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 5 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 5 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 5 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 5 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 5 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 5 },

    // Fusing tests for permute_tile_8x8_4x4
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 5 },

    // Fusing tests for permute_tile_8x8_4x4_fsv16
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 5 },

    // Fusing tests for permute_bfzyx_to_bfyxz
    permute_params{ CASE_PERMUTE_TILE_BFZYX_TO_BFYXZ_0, 2, 5 },

    // Fusing tests for permute_f_y_axes
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_4, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_5, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_6, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_7, 2, 5 },

    // Fusing tests for permute_xy_swap
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_0, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_1, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_2, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_0, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_1, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_S8_0, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_U8_0, 2, 5 }
}));

class permute_quant_u8: public PermuteFusingTest {};
TEST_P(permute_quant_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        permute("permute", input_info("input"), p.permute_order),
        quantize("quant", input_info("permute"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_quant_u8, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 3 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 3 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 3 },
    permute_params{ CASE_PERMUTE_F32_8, 2, 3 },

    // Fusing tests for permute_xy_swap.
    // Note: this suite quantizes the permute output directly to u8; there is no
    // matching quantize kernel for `i8 -> u8` or `u8 -> u8` at these shapes, so
    // int-input cases are intentionally excluded here.
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_0, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_1, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_2, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_1, 2, 3 },
}));

class permute_scale_actv_eltw_scale_actv_quant_i8: public PermuteFusingTest {};
TEST_P(permute_scale_actv_eltw_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale1_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltw_data", get_mem(layout(p.data_type, p.input_format, p.out_shape))),
        data("scale2_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        permute("permute", input_info("input"), p.permute_order),
        eltwise("scale1", { input_info("permute"), input_info("scale1_data") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, p.data_type),
        eltwise("scale2", { input_info("eltw"), input_info("scale2_data") }, eltwise_mode::prod, p.default_type),
        activation("actv2", input_info("scale2"), activation_func::relu),
        quantize("quant", input_info("actv2"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("out", input_info("quant"), p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_actv_eltw_scale_actv_quant_i8, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 8 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 8 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 8 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 8 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 8 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 8 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 8 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 8 },

    // Fusing tests for permute_xy_swap
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_0, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_1, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_2, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_0, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_1, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_S8_0, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_U8_0, 2, 8 },
}));

class permute_scale_eltwise_actv_scale_actv: public PermuteFusingTest {};
TEST_P(permute_scale_eltwise_actv_scale_actv, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape })),
        data("scale_data1", get_mem(get_per_channel_layout(p), 1e-1f)),
        data("scale_data2", get_mem(get_per_channel_layout(p), 1e-1f)),
        permute("permute", input_info("input"), p.permute_order),
        eltwise("scale1", { input_info("permute"), input_info("scale_data1") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        eltwise("eltwise", { input_info("actv1"), input_info("eltwise_data") }, eltwise_mode::sum, p.default_type),
        eltwise("scale2", { input_info("eltwise"), input_info("scale_data2") }, eltwise_mode::prod, p.default_type),
        activation("actv2", input_info("scale2"), activation_func::relu),
        reorder("reorder_bfyx", input_info("actv2"), p.default_format, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_eltwise_actv_scale_actv, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 7 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 7 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 7 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 7 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 7 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 7 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 7 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 7 },

    // Fusing tests for permute_tile_8x8_4x4
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 7 },

    // Fusing tests for permute_tile_8x8_4x4_fsv16
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 7 },

    // Fusing tests for permute_bfzyx_to_bfyxz
    permute_params{ CASE_PERMUTE_TILE_BFZYX_TO_BFYXZ_0, 2, 7 },

    // Fusing tests for permute_f_y_axes
    permute_params{ CASE_PERMUTE_TILE_BFYX_TO_BYFX_0, 2, 7 },

    // Fusing tests for permute_xy_swap
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_0, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_1, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_2, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_0, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_1, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_S8_0, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_U8_0, 2, 7 },
}));

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------- PERMUTE FUSE REDUNDANT REORDER cases ------------------------------------------ */
/* ------------------------------------------------------------------------------------------------------------ */

#define CASE_PERMUTE_REORDER_F32_0 { 1, 16, 32, 2 },   { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::f32, data_types::f32, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F32_1 { 2, 7, 9, 27 },  { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::f32, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_F32_2 { 1, 16, 4, 5, 16 }, { 0, 2, 3, 4, 1 }, { 0, 2, 3, 4, 1 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_3 { 1, 16, 32, 2 },   { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::f32, data_types::f32,  format::bfyx, format::b_fs_yx_fsv16
#define CASE_PERMUTE_REORDER_F16_0 { 1, 16, 2, 4 },     { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F16_1 { 1, 16, 4, 5, 16 }, { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_2 { 1, 5, 1, 2, 14 },  { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_3 { 1, 16, 2, 4 },     { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::f16, data_types::f16,  format::bfyx, format::b_fs_yx_fsv16

// type change
#define CASE_PERMUTE_REORDER_S8_TO_F32_0 { 1, 15, 4, 5 },    { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::i8, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_S8_TO_F32_1 { 1, 2, 15, 4, 5 }, { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_TO_F16_0 { 1, 5, 1, 2, 14 }, { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::f32, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_U8_TO_F16_0 { 1, 17, 1, 2, 7 },  { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::u8, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx

// dim change
#define CASE_PERMUTE_REORDER_4D_TO_5D_F32_0 { 1, 16, 8, 16 }, { 1, 3, 2, 0 }, { 0, 3, 4, 2, 1 }, data_types::f32, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_4D_TO_6D_F32_1 { 1, 16, 8, 16 }, { 0, 3, 1, 2 }, { 0, 4, 5, 1, 3, 2 }, data_types::f32, data_types::f32, format::bfyx, format::bfwzyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_0 { 1, 16, 4, 5, 18 },{ 0, 2, 1, 3, 4 }, { 0, 2, 3, 1 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_1 { 1, 16, 4, 5, 16 },{ 0, 4, 1, 2, 3 }, { 0, 2, 3, 1 }, data_types::f32, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_6D_F32_2 { 1, 16, 8, 4, 16 }, { 0, 2, 1, 3, 4 }, { 0, 4, 5, 1, 3, 2 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx
#define CASE_PERMUTE_REORDER_6D_TO_4D_F32_0 { 1, 16, 4, 5, 4, 16 }, { 0, 5, 1, 4, 3, 2 }, { 0, 2, 3, 1 }, data_types::f32, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_6D_TO_5D_F32_1 { 1, 16, 4, 5, 4, 16 }, { 0, 5, 1, 4, 3, 2 }, { 0, 3, 4, 1, 2 }, data_types::f32, data_types::f32, format::bfwzyx, format::bfzyx

// permute_opt for blocked format
#define CASE_PERMUTE_REORDER_TILED_F32_0 { 1, 256, 2, 64 }, { 0, 2, 3, 1 }, { 0, 3, 1, 2 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_1 { 1, 78, 2, 259 }, { 0, 2, 3, 1 }, { 0, 3, 1, 2 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_2 { 1, 48, 1, 3, 259 }, { 0, 2, 3, 4, 1 }, { 0, 4, 1, 2, 3 },  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx

// permute_opt for blocked format => reorder to differnt dim
#define CASE_PERMUTE_REORDER_TILED_F32_3 { 1, 45, 1, 3, 259 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_4 { 2, 273, 19, 19 }, { 0, 2, 3, 1 }, { 0, 3, 1, 2 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_5 { 2, 546, 2, 2 }, { 0, 2, 3, 1 }, { 0, 3, 1, 2 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx

// permute opt for blocked format => reorder to different dim/type
#define CASE_PERMUTE_REORDER_TILED_I8_4 { 1, 45, 1, 3, 259 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_5 { 1, 48, 3, 256 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2 },  data_types::f16, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_6 { 1, 48, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 5, 1, 4, 3, 2 },  data_types::f16, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx

// permute opt for non_blocked format => reorder to differnt dim/type
#define CASE_PERMUTE_REORDER_TILED_F16_7 { 1, 48, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_8 { 1, 28, 2, 2, 3, 256 }, { 0, 2, 3, 4, 5, 1 }, { 0, 3, 1, 2 },  data_types::f16, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_9 { 1, 24, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_10 { 1, 35, 3, 253 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2 },  data_types::f16, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_11 { 1, 32, 3, 253 }, { 0, 2, 3, 1 }, { 0, 5, 1, 4, 2, 3 },  data_types::f16, data_types::f32, format::bfyx, format::bfwzyx
#define CASE_PERMUTE_REORDER_TILED_F16_12 { 1, 768, 32, 32 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2},  data_types::f16, data_types::f32, format::bfyx, format::bfzyx

class permute_redundant_reorder : public PermuteReorderFusingTest {};
TEST_P(permute_redundant_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", input_info("input"), p.permute_order1),
        reorder("reorder1", input_info("permute1"), p.output_format, p.output_type),    // to be fused
        permute("permute2", input_info("reorder1"), p.permute_order2)                   // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_redundant_reorder, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_1, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_1, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_2, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_1, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_TO_F16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_U8_TO_F16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_3, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_4, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_I8_4, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_5, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_6, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_7, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_8, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_9, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_10, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_11, 3, 4 },
}));

class permute_redundant_reorder_dynamic : public PermuteReorderFusingTest {};
TEST_P(permute_redundant_reorder_dynamic, basic) {
    cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_dynamic_input_layout(p)),
        permute("permute1", input_info("input"), p.permute_order1),
        reorder("reorder1", input_info("permute1"), p.output_format, p.output_type),    // to be fused
        permute("permute2", input_info("reorder1"), p.permute_order2)                   // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_redundant_reorder_dynamic, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_3, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_3, 3, 4 },
}));

class permute_act_reorder : public PermuteReorderFusingTest {};

TEST_P(permute_act_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", input_info("input"), p.permute_order1),
        activation("activation", input_info("permute1"), activation_func::abs),
        reorder("reorder1", input_info("activation"), p.output_format, p.output_type),  // to be fused
        permute("permute2", input_info("reorder1"), p.permute_order2)                   // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_act_reorder, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_0, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_1, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_0, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_1, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_2, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_0, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_1, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_3, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_5, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_6, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_7, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_8, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_9, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_10, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_11, 3, 5 },
}));

class permute_eltwise_reorder : public PermuteReorderFusingTest {};

TEST_P(permute_eltwise_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("elt_data", get_mem(get_elt_input_layout(p))),
        permute("permute1", input_info("input"), p.permute_order1),
        eltwise("elt", { input_info("permute1"), input_info("elt_data") }, eltwise_mode::sum, p.permute_type),
        reorder("reorder1", input_info("elt"), p.output_format, p.output_type),  // to be fused to prev permute
        permute("permute2", input_info("reorder1"), p.permute_order2)            // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

// Tiled opt kernel should not be fused with eltwise + reorder. Currently permute_ref will be selected and fused with eltwise + reorder
INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_eltwise_reorder, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_7, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_8, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_9, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_10, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_11, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_12, 3, 5 },
}));

// -----------------------------------------------------------------------------
// Regression coverage for the fused-eltwise higher-rank-peer canonicalization fix
// in canonicalize_fused_shapes() (kernel_selector_helper.h).
//
// Reproduces the df1 output-0 defect: in legacy shape-inference mode (the
// default), a permute whose 5D output is flattened to 4D bfyx by the layout
// optimizer (because a downstream Reshape consumes it) has an eltwise Add fused
// in as a post-op, while the fused Add's peer dependency stays 5D bfzyx.
// Before the fix, canonicalize_fused_shapes() could not down-rank the higher-rank
// 5D peer to the 4D host output (extend_shape_to_rank_from_begin() only *extends*),
// so the fused eltwise kernel read the 5D peer with 4D indexing and scrambled data
// (MSE ~= 12-20 vs CPU). The fix folds the contiguous planar peer onto the host
// shape, keeping the Add fused and producing the correct result.
//
// This mirrors the accepted reproducer_v3.xml built in-memory with its exact
// f16 weights, so it exercises the real GPU compile/layout/fusion path rather
// than a hand-forced cldnn topology (the 5D->4D flattening is IR-layout-optimizer
// driven and cannot be reproduced with raw cldnn primitives). The reference is the
// same model compiled on the reference device (CPU); GPU must match it. The
// fusion, impl, and 4D-host / 5D-peer layout state are asserted explicitly so a
// silently-unfused or re-laid-out graph cannot make the test pass for the wrong
// reason. A companion NSI variant asserts the rank-consistent (both-5D) path stays
// correct as well. A broadcast variant asserts that a legal higher-rank NumPy
// broadcast peer (fewer elements than the host) compiles by default and matches
// CPU, proving the fold is not restricted to the equal-total reshape case and that
// no valid model fails compilation. Because ov::Core loads the installed GPU
// plugin, validating a source change with these tests requires building AND
// installing the plugin.

#include <cstdlib>

#include <openvino/runtime/core.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/exec_model_info.hpp>
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"

namespace {
// Exact f16 weights of reproducer_v3.bin (branch-1 and branch-2 MatMul, [12,8]).
const std::vector<float> kReproV3Weights1 = {
    -1.085938f, 0.997559f, 0.282959f, -1.505859f, -0.578613f, 1.651367f, -2.425781f, -0.428955f,
    1.265625f, -0.866699f, -0.678711f, -0.094727f, 1.491211f, -0.638672f, -0.444092f, -0.434326f,
    2.205078f, 2.187500f, 1.003906f, 0.386230f, 0.737305f, 1.491211f, -0.936035f, 1.175781f,
    -1.253906f, -0.637695f, 0.907227f, -1.428711f, -0.140015f, -0.861816f, -0.255615f, -2.798828f,
    -1.771484f, -0.699707f, 0.927246f, -0.173584f, 0.002846f, 0.687988f, -0.879395f, 0.283691f,
    -0.805176f, -1.727539f, -0.390869f, 0.573730f, 0.338623f, -0.011833f, 2.392578f, 0.412842f,
    0.978516f, 2.238281f, -1.293945f, -1.039062f, 1.744141f, -0.797852f, 0.029678f, 1.069336f,
    0.890625f, 1.754883f, 1.496094f, 1.069336f, -0.772949f, 0.794922f, 0.314209f, -1.326172f,
    1.416992f, 0.807129f, 0.045502f, -0.233032f, -1.198242f, 0.199585f, 0.468506f, -0.831055f,
    1.162109f, -1.097656f, -2.123047f, 1.040039f, -0.403320f, -0.125977f, -0.837402f, -1.606445f,
    1.254883f, -0.688965f, 1.661133f, 0.807129f, -0.314697f, -1.085938f, -0.732422f, -1.212891f,
    2.087891f, 0.164429f, 1.150391f, -1.267578f, 0.181030f, 1.177734f, -0.334961f, 1.031250f};

const std::vector<float> kReproV3Weights2 = {
    -1.084961f, -1.363281f, 0.379395f, -0.379150f, 0.642090f, -1.977539f, 0.712402f, 2.597656f,
    -0.024628f, 0.034149f, 0.179565f, -1.862305f, 0.426025f, -1.605469f, -0.427734f, 1.243164f,
    -0.735352f, 0.501465f, 1.012695f, 0.278809f, -1.371094f, -0.332520f, 1.958984f, -2.025391f,
    -0.275879f, -0.552246f, 0.120728f, 0.748047f, 1.608398f, -0.270264f, 0.812500f, 0.499756f,
    0.474365f, -0.563965f, -0.997559f, -1.099609f, -0.756348f, 0.321777f, 0.760742f, 0.323486f,
    -0.548828f, 1.805664f, 1.518555f, -0.354004f, -0.823242f, 0.130249f, 1.267578f, 0.332764f,
    0.556641f, -0.212036f, 0.456299f, 1.544922f, -0.239624f, 0.143311f, 0.253906f, 0.283691f,
    -1.412109f, -1.876953f, -1.019531f, 0.167969f, 0.553711f, -0.530762f, 1.376953f, -0.143188f,
    0.020309f, -0.193970f, 0.134033f, 0.704590f, 0.665527f, -0.898438f, 1.523438f, -1.094727f,
    0.079224f, -0.274414f, -1.048828f, -0.075134f, -0.740723f, 0.072937f, 0.403076f, 1.471680f,
    0.307373f, -0.611328f, -0.391602f, 0.140015f, 0.093445f, 1.459961f, 1.395508f, -0.358887f,
    -0.548828f, -2.556641f, -0.548828f, -0.978027f, -0.354736f, 0.391602f, 0.177246f, -0.029968f};

// Builds the reproducer_v3 model:
//   in[1,2,60,12] -> Reshape[1,2,6,10,12] -> MatMul(*W) -> [1,2,6,10,8]
//     -> Transpose[0,1,4,2,3] -> [1,2,8,6,10]     (two such branches)
//   Add(branch1, branch2) -> Reshape[2,8,6,10] -> Result
std::shared_ptr<ov::Model> build_repro_v3_model() {
    using namespace ov;
    auto make_branch = [](const std::shared_ptr<op::v0::Parameter>& param,
                          const std::vector<float>& w) {
        auto to5d = op::v0::Constant::create(element::i64, Shape{5}, {1, 2, 6, 10, 12});
        auto reshape5d = std::make_shared<op::v1::Reshape>(param, to5d, false);
        auto weights = std::make_shared<op::v0::Constant>(element::f16, Shape{12, 8},
                                                          std::vector<ov::float16>(w.begin(), w.end()));
        auto matmul = std::make_shared<op::v0::MatMul>(reshape5d, weights, false, false);
        auto order = op::v0::Constant::create(element::i64, Shape{5}, {0, 1, 4, 2, 3});
        return std::make_shared<op::v1::Transpose>(matmul, order);
    };

    auto in1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 2, 60, 12});
    auto in2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 2, 60, 12});
    in1->set_friendly_name("input1");
    in2->set_friendly_name("input2");

    auto t1 = make_branch(in1, kReproV3Weights1);
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, kReproV3Weights2);
    t2->set_friendly_name("Transpose_peer");

    auto add = std::make_shared<op::v1::Add>(t1, t2);
    add->set_friendly_name("Add_target");

    auto to4d = op::v0::Constant::create(element::i64, Shape{4}, {2, 8, 6, 10});
    auto reshape4d = std::make_shared<op::v1::Reshape>(add, to4d, false);
    reshape4d->set_friendly_name("Reshape_to4D");

    auto result = std::make_shared<op::v0::Result>(reshape4d);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{in1, in2}, "reproducer_v3");
}

// Compiles build_repro_v3_model() on the given device and returns the f16 output flattened to float.
std::pair<std::vector<float>, ov::CompiledModel> run_repro_v3(ov::Core& core,
                                                              const std::string& device,
                                                              const ov::AnyMap& cfg,
                                                              const ov::Tensor& in1,
                                                              const ov::Tensor& in2) {
    auto model = build_repro_v3_model();
    auto compiled = core.compile_model(model, device, cfg);
    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, in1);
    req.set_input_tensor(1, in2);
    req.infer();
    auto out = req.get_output_tensor(0);
    std::vector<float> vals(out.get_size());
    const auto* p = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i)
        vals[i] = static_cast<float>(p[i]);
    return std::make_pair(vals, compiled);
}

double repro_v3_mse(const std::vector<float>& a, const std::vector<float>& b, double& max_ae) {
    double se = 0.0;
    max_ae = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        se += d * d;
        max_ae = std::max(max_ae, std::abs(d));
    }
    return se / static_cast<double>(a.size());
}

bool discover_gpu_and_cpu(ov::Core& core) {
    std::vector<std::string> devices;
    try {
        devices = core.get_available_devices();
    } catch (...) {}
    const bool has_gpu = std::any_of(devices.begin(), devices.end(),
                                     [](const std::string& d) { return d.rfind("GPU", 0) == 0; });
    const bool has_cpu = std::any_of(devices.begin(), devices.end(),
                                     [](const std::string& d) { return d == "CPU"; });
    return has_gpu && has_cpu;
}

void fill_repro_v3_inputs(ov::Tensor& in1, ov::Tensor& in2) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto rnd = rg.generate_random_1d<ov::float16>(1 * 2 * 60 * 12, -2, 2);
    std::copy(rnd.begin(), rnd.end(), in1.data<ov::float16>());
    auto rnd2 = rg.generate_random_1d<ov::float16>(1 * 2 * 60 * 12, -2, 2);
    std::copy(rnd2.begin(), rnd2.end(), in2.data<ov::float16>());
}

// Cross-platform, exception-safe scoped environment-variable override. On construction it records the
// previous value (if any) and sets the new one; on destruction it restores the previous value or unsets
// the variable, so every exit path (including a thrown/failed assertion) leaves the environment as it
// was found. Uses _putenv_s on Windows and setenv/unsetenv elsewhere.
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : m_name(name) {
        const char* prev = std::getenv(name);
        if (prev != nullptr) {
            m_had_prev = true;
            m_prev = prev;
        }
        set(name, value);
    }
    ~ScopedEnvVar() {
        if (m_had_prev) {
            set(m_name.c_str(), m_prev.c_str());
        } else {
            unset(m_name.c_str());
        }
    }
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    static void set(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        ::setenv(name, value, 1);
#endif
    }
    static void unset(const char* name) {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        ::unsetenv(name);
#endif
    }
    std::string m_name;
    std::string m_prev;
    bool m_had_prev = false;
};

// Builds a higher-rank *broadcast* variant of the reproducer_v3 model: the host branch produces
// [1,2,8,6,10] (flattened to 4D bfyx [1,2,48,10]) while the peer branch produces [1,1,8,6,10] (5D
// bfzyx), which broadcasts against the host over the feature dim. Element counts differ (960 vs 480),
// so this is a legal NumPy broadcast, NOT the equal-total reshape of reproducer_v3. Before the Worker
// 08R fix this aborted GPU compilation at the Worker 08 fold assertion; the fix must compile it and
// match CPU with the Add kept fused.
std::shared_ptr<ov::Model> build_broadcast_repro_model() {
    using namespace ov;
    auto make_branch = [](const std::shared_ptr<op::v0::Parameter>& param, int64_t f,
                          const std::vector<float>& w) {
        auto to5d = op::v0::Constant::create(element::i64, Shape{5},
                                             std::vector<int64_t>{1, f, 6, 10, 12});
        auto reshape5d = std::make_shared<op::v1::Reshape>(param, to5d, false);
        auto weights = std::make_shared<op::v0::Constant>(element::f16, Shape{12, 8},
                                                          std::vector<ov::float16>(w.begin(), w.end()));
        auto matmul = std::make_shared<op::v0::MatMul>(reshape5d, weights, false, false);
        auto order = op::v0::Constant::create(element::i64, Shape{5}, {0, 1, 4, 2, 3});
        return std::make_shared<op::v1::Transpose>(matmul, order);
    };

    auto in1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 2, 60, 12});
    auto in2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, 60, 12});
    in1->set_friendly_name("input1");
    in2->set_friendly_name("input2");

    auto t1 = make_branch(in1, 2, kReproV3Weights1);  // host: [1,2,8,6,10]
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, 1, kReproV3Weights2);  // peer: [1,1,8,6,10] (broadcasts over F)
    t2->set_friendly_name("Transpose_peer");

    auto add = std::make_shared<op::v1::Add>(t1, t2);
    add->set_friendly_name("Add_target");

    auto to4d = op::v0::Constant::create(element::i64, Shape{4}, {2, 8, 6, 10});
    auto reshape4d = std::make_shared<op::v1::Reshape>(add, to4d, false);
    reshape4d->set_friendly_name("Reshape_to4D");

    auto result = std::make_shared<op::v0::Result>(reshape4d);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{in1, in2}, "broadcast_repro");
}

std::pair<std::vector<float>, ov::CompiledModel> run_broadcast_repro(ov::Core& core,
                                                                     const std::string& device,
                                                                     const ov::Tensor& in1,
                                                                     const ov::Tensor& in2) {
    auto model = build_broadcast_repro_model();
    auto compiled = core.compile_model(model, device);
    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, in1);
    req.set_input_tensor(1, in2);
    req.infer();
    auto out = req.get_output_tensor(0);
    std::vector<float> vals(out.get_size());
    const auto* p = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i)
        vals[i] = static_cast<float>(p[i]);
    return std::make_pair(vals, compiled);
}
}  // namespace

// Default (legacy) path: the target permute host output is flattened to 4D bfyx while the fused Add's
// peer dependency stays 5D bfzyx. The fix must keep the Add fused, keep permute_ref, and produce a
// result that matches the CPU reference (the higher-rank peer is folded onto the 4D host shape).
TEST(permute_fused_eltwise_rank_mismatch, legacy_5d_peer_into_4d_host) {
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, 2, 60, 12});
    fill_repro_v3_inputs(in1, in2);

    auto [cpu_vals, cpu_compiled] = run_repro_v3(core, "CPU", {}, in1, in2);
    auto [gpu_vals, gpu_compiled] = run_repro_v3(core, "GPU", {}, in1, in2);

    // --- Prove the buggy graph state is actually present on GPU (4D fused host + 5D peer). ---
    auto rt = gpu_compiled.get_runtime_model();
    bool found_fused_target = false;
    bool target_is_4d = false;
    bool target_is_permute_ref = false;
    bool peer_is_5d = false;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        std::string impl;
        if (auto i2 = info.find(ov::exec_model_info::IMPL_TYPE); i2 != info.end())
            impl = i2->second.as<std::string>();

        // Target permute: Add fused in, permute_ref, 4D output.
        if (orig.find("Transpose_target") != std::string::npos &&
            orig.find("Add_target") != std::string::npos) {
            found_fused_target = true;
            target_is_permute_ref = impl.find("permute_ref") != std::string::npos;
            target_is_4d = node->get_output_partial_shape(0).rank().get_length() == 4;
        }
        // Peer permute: still 5D.
        if (orig.find("Transpose_peer") != std::string::npos &&
            orig.find("Add_target") == std::string::npos) {
            peer_is_5d = node->get_output_partial_shape(0).rank().get_length() == 5;
        }
    }
    ASSERT_TRUE(found_fused_target) << "Add_target was not fused into Transpose_target on GPU; "
                                       "the fusion mechanism under test is absent.";
    ASSERT_TRUE(target_is_permute_ref) << "Target permute did not select permute_ref.";
    ASSERT_TRUE(target_is_4d) << "Target permute host output was not flattened to 4D.";
    ASSERT_TRUE(peer_is_5d) << "Fused peer dependency was not 5D; rank-mismatch condition absent.";

    // --- Numerical gate: GPU must match the CPU reference. ---
    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    double max_ae = 0.0;
    double mse = repro_v3_mse(gpu_vals, cpu_vals, max_ae);
    // FP16 tolerance: correct path agrees with CPU to ~1e-3 MSE; the bug produced MSE ~12.
    EXPECT_LT(mse, 1e-2) << "GPU fused-permute output diverges from CPU reference. "
                            "MSE=" << mse << " MaxAbsErr=" << max_ae
                         << " (rank-mismatched fused eltwise reads 5D peer with 4D indexing).";
}

// New-shape-infer (NSI) path: the host output stays 5D bfzyx so the fused peer is already
// rank-consistent (both 5D). This path never entered the broken repair branch and must remain correct;
// it guards against a fix that only works because it changed the default flattening behavior.
// NSI is a RELEASE_INTERNAL option, so it is selected through its OV_GPU environment variable (the same
// mechanism used by the standalone reproducer controls), not via a public compile_model property.
TEST(permute_fused_eltwise_rank_mismatch, nsi_5d_peer_and_5d_host) {
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, 2, 60, 12});
    fill_repro_v3_inputs(in1, in2);

    auto [cpu_vals, cpu_compiled] = run_repro_v3(core, "CPU", {}, in1, in2);

    // Use a fresh Core so the NSI env option is picked up for the GPU compile. The scoped guard restores
    // any pre-existing value on every exit path, including a failed assertion below.
    std::pair<std::vector<float>, ov::CompiledModel> gpu_result;
    {
        ScopedEnvVar nsi_env("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
        ov::Core nsi_core;
        gpu_result = run_repro_v3(nsi_core, "GPU", {}, in1, in2);
    }
    auto& gpu_vals = gpu_result.first;
    auto& gpu_compiled = gpu_result.second;

    // Under NSI the fused target permute output stays 5D (rank-consistent with the peer).
    auto rt = gpu_compiled.get_runtime_model();
    bool found_fused_target = false;
    bool target_is_5d = false;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        if (orig.find("Transpose_target") != std::string::npos &&
            orig.find("Add_target") != std::string::npos) {
            found_fused_target = true;
            target_is_5d = node->get_output_partial_shape(0).rank().get_length() == 5;
        }
    }
    ASSERT_TRUE(found_fused_target) << "Add_target was not fused into Transpose_target under NSI.";
    ASSERT_TRUE(target_is_5d) << "Under NSI the fused permute host output should remain 5D.";

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    double max_ae = 0.0;
    double mse = repro_v3_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU (NSI) fused-permute output diverges from CPU reference. "
                            "MSE=" << mse << " MaxAbsErr=" << max_ae;
}

// Legal higher-rank NumPy broadcast peer: the host branch produces a 4D-flattened bfyx [1,2,48,10]
// host while the fused peer stays 5D bfzyx [1,1,8,6,10] and broadcasts over the feature dim. The peer
// holds FEWER elements than the host (480 vs 960), so it is not the equal-total reshape case; Worker
// 08's equal-total predicate rejected it and its OPENVINO_ASSERT aborted compilation of this valid
// model. The Worker 08R fix must compile it by default (no override), keep the Add fused, and match the
// CPU reference. This is the primary regression guard for the rejected-broadcast defect.
TEST(permute_fused_eltwise_rank_mismatch, legacy_higher_rank_broadcast_peer_compiles_and_matches_cpu) {
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, 1, 60, 12});
    auto rnd1 = rg.generate_random_1d<ov::float16>(1 * 2 * 60 * 12, -2, 2);
    std::copy(rnd1.begin(), rnd1.end(), in1.data<ov::float16>());
    auto rnd2 = rg.generate_random_1d<ov::float16>(1 * 1 * 60 * 12, -2, 2);
    std::copy(rnd2.begin(), rnd2.end(), in2.data<ov::float16>());

    auto [cpu_vals, cpu_compiled] = run_broadcast_repro(core, "CPU", in1, in2);

    // Default GPU compilation must SUCCEED (the Worker 08 assertion is removed) and keep the Add fused.
    std::vector<float> gpu_vals;
    ov::CompiledModel gpu_compiled;
    ASSERT_NO_THROW({
        auto res = run_broadcast_repro(core, "GPU", in1, in2);
        gpu_vals = res.first;
        gpu_compiled = res.second;
    }) << "Default GPU compilation of the valid higher-rank broadcast model must not fail.";

    // The legal Add fusion must be retained (the fold represents the peer at the host rank).
    auto rt = gpu_compiled.get_runtime_model();
    bool found_fused_target = false;
    bool target_is_permute_ref = false;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        std::string impl;
        if (auto i2 = info.find(ov::exec_model_info::IMPL_TYPE); i2 != info.end())
            impl = i2->second.as<std::string>();
        if (orig.find("Transpose_target") != std::string::npos &&
            orig.find("Add_target") != std::string::npos) {
            found_fused_target = true;
            target_is_permute_ref = impl.find("permute_ref") != std::string::npos;
        }
    }
    ASSERT_TRUE(found_fused_target) << "Add_target was not fused into Transpose_target on GPU; the "
                                       "broadcast fold must retain the legal fusion.";
    EXPECT_TRUE(target_is_permute_ref) << "Target permute did not select permute_ref.";

    // Numerical gate: GPU broadcast result must match the CPU reference.
    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    double max_ae = 0.0;
    double mse = repro_v3_mse(gpu_vals, cpu_vals, max_ae);
    for (float v : gpu_vals)
        ASSERT_TRUE(std::isfinite(v)) << "GPU broadcast output has non-finite values.";
    EXPECT_LT(mse, 1e-2) << "GPU higher-rank-broadcast fused-permute output diverges from CPU. "
                            "MSE=" << mse << " MaxAbsErr=" << max_ae;
}

// -----------------------------------------------------------------------------
// Worker 08S: behavior-level collapsed-axis broadcast matrix.
//
// Host permute output is 5D bfzyx [1,2,8,6,10], flattened by the layout optimizer
// to 4D bfyx [1,2,48,10] (collapsing z=8,y=6 -> 48) because a downstream Reshape
// consumes it. The fused Add peer keeps its 5D rank with a per-axis broadcast mask
// over {f,z,y,x}. The direct fix retains rank reduction only when the shared
// fold_higher_rank_fused_peer() proof can represent that peer at the actual 4D
// output layout. Every other mask conservatively keeps the permute at 5D, avoiding
// the mismatched indexing that made inner-spatial broadcasts silently wrong. Every
// mask below must compile by default on GPU.1, be finite, and match the CPU
// reference; the intended fused/rank-preserved runtime state is asserted so a
// silently-unfused or mis-laid-out graph cannot pass for the wrong reason.
namespace {

struct collapse_mask_case {
    // peer extents on f,z,y,x (1 => broadcast). Host is always [1,2,8,6,10].
    int64_t pf, pz, py, px;
    // Expected: does the host permute stay higher-rank because no exact 4D peer
    // representation can be proven?
    bool expect_rank_preserved;
    const char* label;
};

// Builds host [1,2,8,6,10] + peer [1,pf,pz,py,px], Add (numpy broadcast), Reshape to 4D.
std::shared_ptr<ov::Model> build_collapse_mask_model(const collapse_mask_case& c) {
    using namespace ov;
    auto make_branch = [](const std::shared_ptr<op::v0::Parameter>& param, int64_t f, int64_t z, int64_t y, int64_t x, const std::vector<float>& w) {
        // pre-transpose logical [1,f,y,x,K] -> matmul(K,z) -> [1,f,y,x,z] -> transpose[0,1,4,2,3]
        const int64_t K = 12;
        auto to5d = op::v0::Constant::create(element::i64, Shape{5}, std::vector<int64_t>{1, f, y, x, K});
        auto reshape5d = std::make_shared<op::v1::Reshape>(param, to5d, false);
        std::vector<ov::float16> wz(static_cast<size_t>(K * z));
        for (size_t i = 0; i < wz.size(); ++i)
            wz[i] = static_cast<ov::float16>(w[i % w.size()]);
        auto weights = std::make_shared<op::v0::Constant>(element::f16, Shape{static_cast<size_t>(K), static_cast<size_t>(z)}, wz);
        auto matmul = std::make_shared<op::v0::MatMul>(reshape5d, weights, false, false);
        auto order = op::v0::Constant::create(element::i64, Shape{5}, {0, 1, 4, 2, 3});
        return std::make_shared<op::v1::Transpose>(matmul, order);  // [1,f,z,y,x]
    };

    auto in1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 2, 60, 12});
    auto in2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, c.pf, c.py * c.px, 12});
    in1->set_friendly_name("input1");
    in2->set_friendly_name("input2");

    auto t1 = make_branch(in1, 2, 8, 6, 10, kReproV3Weights1);  // host [1,2,8,6,10]
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, c.pf, c.pz, c.py, c.px, kReproV3Weights2);  // peer [1,pf,pz,py,px]
    t2->set_friendly_name("Transpose_peer");

    auto add = std::make_shared<op::v1::Add>(t1, t2);
    add->set_friendly_name("Add_target");

    auto to4d = op::v0::Constant::create(element::i64, Shape{4}, {2, 8, 6, 10});
    auto reshape4d = std::make_shared<op::v1::Reshape>(add, to4d, false);
    reshape4d->set_friendly_name("Reshape_to4D");

    auto result = std::make_shared<op::v0::Result>(reshape4d);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{in1, in2}, "collapse_mask");
}

std::pair<std::vector<float>, ov::CompiledModel> run_collapse_mask(ov::Core& core,
                                                                   const std::string& device,
                                                                   const collapse_mask_case& c,
                                                                   const ov::Tensor& in1,
                                                                   const ov::Tensor& in2) {
    auto model = build_collapse_mask_model(c);
    auto compiled = core.compile_model(model, device);
    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, in1);
    req.set_input_tensor(1, in2);
    req.infer();
    auto out = req.get_output_tensor(0);
    std::vector<float> vals(out.get_size());
    const auto* p = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i)
        vals[i] = static_cast<float>(p[i]);
    return std::make_pair(vals, compiled);
}

class permute_fused_collapse_broadcast_matrix : public ::testing::TestWithParam<collapse_mask_case> {};

}  // namespace

TEST_P(permute_fused_collapse_broadcast_matrix, compiles_finite_and_matches_cpu) {
    const auto c = GetParam();
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, static_cast<size_t>(c.pf), static_cast<size_t>(c.py * c.px), 12});
    auto rnd1 = rg.generate_random_1d<ov::float16>(in1.get_size(), -2, 2);
    std::copy(rnd1.begin(), rnd1.end(), in1.data<ov::float16>());
    auto rnd2 = rg.generate_random_1d<ov::float16>(in2.get_size(), -2, 2);
    std::copy(rnd2.begin(), rnd2.end(), in2.data<ov::float16>());

    auto [cpu_vals, cpu_compiled] = run_collapse_mask(core, "CPU", c, in1, in2);

    std::vector<float> gpu_vals;
    ov::CompiledModel gpu_compiled;
    ASSERT_NO_THROW({
        auto res = run_collapse_mask(core, "GPU", c, in1, in2);
        gpu_vals = res.first;
        gpu_compiled = res.second;
    }) << "Default GPU compilation must not fail for mask "
       << c.label;

    // Assert the intended runtime state: Add fused into Transpose_target, and the host permute rank
    // matches the fix's decision (5D preserved for the inner-spatial masks, 4D flattened otherwise).
    auto rt = gpu_compiled.get_runtime_model();
    bool found_fused_target = false;
    int64_t target_rank = 0;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        if (orig.find("Transpose_target") != std::string::npos && orig.find("Add_target") != std::string::npos) {
            found_fused_target = true;
            target_rank = node->get_output_partial_shape(0).rank().get_length();
        }
    }
    ASSERT_TRUE(found_fused_target) << "Add_target not fused into Transpose_target for mask " << c.label;
    if (c.expect_rank_preserved) {
        EXPECT_EQ(target_rank, 5) << "Inner-spatial mask " << c.label << " must keep the fused permute host at 5D (rank preserved).";
    } else {
        EXPECT_EQ(target_rank, 4) << "Representable mask " << c.label << " should keep the flattened 4D fused host.";
    }

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    for (float v : gpu_vals)
        ASSERT_TRUE(std::isfinite(v)) << "Non-finite GPU output for mask " << c.label;
    double max_ae = 0.0;
    double mse = repro_v3_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU vs CPU mismatch for mask " << c.label << " MSE=" << mse << " MaxAbsErr=" << max_ae;
}

INSTANTIATE_TEST_SUITE_P(collapse_broadcast_matrix,
                         permute_fused_collapse_broadcast_matrix,
                         ::testing::Values(
                             // host [1,2,8,6,10]; peer f,z,y,x extents (1 == broadcast).
                             collapse_mask_case{2, 8, 6, 10, false, "equal_total"},            // [1,2,8,6,10]
                             collapse_mask_case{1, 8, 6, 10, false, "feature_broadcast"},      // [1,1,8,6,10] (W08R)
                             collapse_mask_case{2, 1, 6, 10, true, "outer_z_broadcast"},       // [1,2,1,6,10]
                             collapse_mask_case{2, 8, 1, 10, true, "inner_y_broadcast"},       // [1,2,8,1,10] REGRESSION
                             collapse_mask_case{2, 1, 1, 10, false, "all_spatial_broadcast"},  // [1,2,1,1,10]
                             collapse_mask_case{2, 8, 6, 1, false, "trailing_x_broadcast"},    // [1,2,8,6,1]
                             collapse_mask_case{2, 8, 1, 1, true, "inner_y_and_x_broadcast"}   // [1,2,8,1,1] REGRESSION
                             ),
                         [](const ::testing::TestParamInfo<collapse_mask_case>& info) {
                             return std::string(info.param.label);
                         });

// -----------------------------------------------------------------------------
// Direct-fix 6D-to-4D matrix. Host permute output is 6D bfwzyx
// [1,2,X=2,W=4,Z=3,Y=5], and a downstream Reshape introduces the actual reduced
// bfyx layout selected by the optimizer. All 16 spatial broadcast masks are
// covered. Rank reduction is retained only when fold_higher_rank_fused_peer()
// proves that the peer can be represented at that exact reduced layout; every
// other mask conservatively keeps the producer at 6D. Each case must remain
// fused, compile by default, be finite, match CPU, and use the expected rank.
namespace {

struct collapse6d_case {
    int64_t pf, px, pw, pz, py;  // peer extents on x,w,z,y (1 => broadcast); host [1,2,2,4,3,5]
    bool expect_rank_preserved;
    const char* label;
};

std::shared_ptr<ov::Model> build_collapse6d_model(const collapse6d_case& c) {
    using namespace ov;
    const int64_t X = 2, K = 12;
    auto make_branch =
        [&](const std::shared_ptr<op::v0::Parameter>& param, int64_t f, int64_t x, int64_t w, int64_t z, int64_t y, const std::vector<float>& wsrc) {
            // pre [1,f,w,z,y,K] -> matmul(K,x) -> [1,f,w,z,y,x] -> transpose[0,1,5,2,3,4] -> [1,f,x,w,z,y]
            auto to6d = op::v0::Constant::create(element::i64, Shape{6}, std::vector<int64_t>{1, f, w, z, y, K});
            auto reshape6d = std::make_shared<op::v1::Reshape>(param, to6d, false);
            std::vector<ov::float16> wx(static_cast<size_t>(K * x));
            for (size_t i = 0; i < wx.size(); ++i)
                wx[i] = static_cast<ov::float16>(wsrc[i % wsrc.size()]);
            auto weights = std::make_shared<op::v0::Constant>(element::f16, Shape{static_cast<size_t>(K), static_cast<size_t>(x)}, wx);
            auto matmul = std::make_shared<op::v0::MatMul>(reshape6d, weights, false, false);
            auto order = op::v0::Constant::create(element::i64, Shape{6}, {0, 1, 5, 2, 3, 4});
            return std::make_shared<op::v1::Transpose>(matmul, order);  // [1,f,x,w,z,y]
        };

    auto in1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 2, 4 * 3 * 5, K});
    auto in2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, c.pf, c.pw * c.pz * c.py, K});
    in1->set_friendly_name("input1");
    in2->set_friendly_name("input2");

    auto t1 = make_branch(in1, 2, X, 4, 3, 5, kReproV3Weights1);  // host [1,2,X,4,3,5]
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, c.pf, c.px, c.pw, c.pz, c.py, kReproV3Weights2);
    t2->set_friendly_name("Transpose_peer");

    auto add = std::make_shared<op::v1::Add>(t1, t2);
    add->set_friendly_name("Add_target");

    // Flatten to 4D [2, X, W, Z*Y].
    auto to4d = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{2, X, 4, 15});
    auto reshape4d = std::make_shared<op::v1::Reshape>(add, to4d, false);
    reshape4d->set_friendly_name("Reshape_to4D");

    auto result = std::make_shared<op::v0::Result>(reshape4d);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{in1, in2}, "collapse6d");
}

std::pair<std::vector<float>, ov::CompiledModel> run_collapse6d(ov::Core& core,
                                                                const std::string& device,
                                                                const collapse6d_case& c,
                                                                const ov::Tensor& in1,
                                                                const ov::Tensor& in2) {
    auto model = build_collapse6d_model(c);
    auto compiled = core.compile_model(model, device);
    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, in1);
    req.set_input_tensor(1, in2);
    req.infer();
    auto out = req.get_output_tensor(0);
    std::vector<float> vals(out.get_size());
    const auto* p = out.data<ov::float16>();
    for (size_t i = 0; i < out.get_size(); ++i)
        vals[i] = static_cast<float>(p[i]);
    return std::make_pair(vals, compiled);
}

class permute_fused_collapse6d_matrix : public ::testing::TestWithParam<collapse6d_case> {};

}  // namespace

TEST_P(permute_fused_collapse6d_matrix, compiles_finite_and_matches_cpu) {
    const auto c = GetParam();
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 4 * 3 * 5, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, static_cast<size_t>(c.pf), static_cast<size_t>(c.pw * c.pz * c.py), 12});
    auto rnd1 = rg.generate_random_1d<ov::float16>(in1.get_size(), -2, 2);
    std::copy(rnd1.begin(), rnd1.end(), in1.data<ov::float16>());
    auto rnd2 = rg.generate_random_1d<ov::float16>(in2.get_size(), -2, 2);
    std::copy(rnd2.begin(), rnd2.end(), in2.data<ov::float16>());

    auto [cpu_vals, cpu_compiled] = run_collapse6d(core, "CPU", c, in1, in2);

    std::vector<float> gpu_vals;
    ov::CompiledModel gpu_compiled;
    ASSERT_NO_THROW({
        auto res = run_collapse6d(core, "GPU", c, in1, in2);
        gpu_vals = res.first;
        gpu_compiled = res.second;
    }) << "Default GPU compilation must not fail for 6D mask "
       << c.label;

    auto rt = gpu_compiled.get_runtime_model();
    bool found_fused_target = false;
    int64_t target_rank = 0;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        if (orig.find("Transpose_target") != std::string::npos && orig.find("Add_target") != std::string::npos) {
            found_fused_target = true;
            target_rank = node->get_output_partial_shape(0).rank().get_length();
        }
    }
    ASSERT_TRUE(found_fused_target) << "Add_target not fused into Transpose_target for 6D mask " << c.label;
    if (c.expect_rank_preserved) {
        EXPECT_EQ(target_rank, 6) << "Inner-spatial 6D mask " << c.label << " must preserve 6D host rank.";
    } else {
        EXPECT_EQ(target_rank, 4) << "Representable 6D mask " << c.label << " should keep the 4D host.";
    }

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    for (float v : gpu_vals)
        ASSERT_TRUE(std::isfinite(v)) << "Non-finite GPU output for 6D mask " << c.label;
    double max_ae = 0.0;
    double mse = repro_v3_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU vs CPU mismatch for 6D mask " << c.label << " MSE=" << mse << " MaxAbsErr=" << max_ae;
}

INSTANTIATE_TEST_SUITE_P(collapse6d_matrix,
                         permute_fused_collapse6d_matrix,
                         ::testing::Values(
                             // host [1,2,X=2,W=4,Z=3,Y=5]. Peer f,x,w,z,y (1 == broadcast).
                             // A 4D metadata fold is retained only when the peer can be proven broadcast-compatible with
                             // the actual reduced layout. Every other mask conservatively preserves the 6D host rank.
                             collapse6d_case{2, 2, 4, 3, 5, false, "equal_total_6d"},
                             collapse6d_case{2, 2, 4, 3, 1, false, "y_broadcast_6d"},
                             collapse6d_case{2, 2, 4, 1, 5, true, "z_broadcast_6d"},
                             collapse6d_case{2, 2, 4, 1, 1, true, "zy_broadcast_6d"},
                             collapse6d_case{2, 2, 1, 3, 5, true, "w_broadcast_6d"},
                             collapse6d_case{2, 2, 1, 3, 1, true, "wy_broadcast_6d"},
                             collapse6d_case{2, 2, 1, 1, 5, true, "wz_broadcast_6d"},
                             collapse6d_case{2, 2, 1, 1, 1, true, "wzy_broadcast_6d"},
                             collapse6d_case{2, 1, 4, 3, 5, true, "x_broadcast_6d"},
                             collapse6d_case{2, 1, 4, 3, 1, true, "xy_broadcast_6d"},
                             collapse6d_case{2, 1, 4, 1, 5, true, "xz_broadcast_6d"},
                             collapse6d_case{2, 1, 4, 1, 1, true, "xzy_broadcast_6d"},
                             collapse6d_case{2, 1, 1, 3, 5, true, "xw_broadcast_6d"},
                             collapse6d_case{2, 1, 1, 3, 1, true, "xwy_broadcast_6d"},
                             collapse6d_case{2, 1, 1, 1, 5, false, "xwz_broadcast_6d"},
                             collapse6d_case{2, 1, 1, 1, 1, false, "xwzy_broadcast_6d"}),
                         [](const ::testing::TestParamInfo<collapse6d_case>& info) {
                             return std::string(info.param.label);
                         });
