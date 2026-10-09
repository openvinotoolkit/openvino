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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <openvino/runtime/core.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/exec_model_info.hpp>
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"

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

#define CASE_PERMUTE_BF16_0 { 1, 16, 4, 5 }, { 1, 16, 4, 5 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::bf16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_1 { 2, 16, 4, 5 }, { 16, 4, 5, 2 }, { 1, 3, 0, 2 }, tensor{ 0 }, data_types::bf16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_2 { 1, 32, 2, 3 }, { 2, 3, 32, 1 }, { 3, 2, 0, 1 }, tensor{ 0 }, data_types::bf16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_3 { 3, 16, 1, 1 }, { 1, 1, 16, 3 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::bf16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_4 { 2, 15, 4, 5 }, { 4, 2, 5, 15 }, { 3, 0, 1, 2 }, tensor{ 0 }, data_types::bf16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_5 { 1, 15, 1, 2 }, { 15, 2, 1, 1 }, { 1, 2, 0, 3 }, tensor{ 0 }, data_types::bf16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_BF16_6 { 1, 15, 4, 4 }, { 4, 4, 1, 15 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::bf16, format::bfyx, data_types::f32, format::bfyx

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

#define CASE_PERMUTE_BF16_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::bf16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_BF16_3D_1 { 2, 15, 4, 3, 4 }, { 4, 4, 2, 15, 3 }, { 4, 2, 3, 1, 0 }, tensor{ 0 }, data_types::bf16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_BF16_3D_2 { 2, 16, 4, 4, 3 }, { 2, 4, 3, 16, 4 }, { 0, 3, 4, 1, 2 }, tensor{ 0 }, data_types::bf16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_BF16_3D_3 { 1, 32, 4, 2, 1 }, { 2, 32, 4, 1, 1 }, { 3, 1, 0, 2, 4 }, tensor{ 0 }, data_types::bf16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_BF16_3D_4 { 16, 16, 1, 1, 1 },{ 1, 16, 1, 1, 16 },{ 2, 0, 1, 4, 3 }, tensor{ 0 }, data_types::bf16, format::bfzyx, data_types::f32, format::bfzyx

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
#define CASE_PERMUTE_XY_SWAP_BF16_0 { 1, 8, 16, 16 }, { 1, 8, 16, 16 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::bf16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_XY_SWAP_BF16_1 { 1, 4, 32, 32 }, { 1, 4, 32, 32 }, { 0, 1, 3, 2 }, tensor{ 0 }, data_types::bf16, format::bfyx, data_types::f32, format::bfyx
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

    permute_params{ CASE_PERMUTE_BF16_0, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_1, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_2, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_3, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_4, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_5, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_6, 2, 5 },

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

    permute_params{ CASE_PERMUTE_BF16_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_3D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_BF16_3D_4, 2, 5 },

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
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_0, 2, 5 },
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_1, 2, 5 },
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
    permute_params{ CASE_PERMUTE_BF16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_BF16_1, 2, 3 },

    // Fusing tests for permute_xy_swap.
    // Note: this suite quantizes the permute output directly to u8; there is no
    // matching quantize kernel for `i8 -> u8` or `u8 -> u8` at these shapes, so
    // int-input cases are intentionally excluded here.
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_0, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_1, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F32_2, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_F16_1, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_1, 2, 3 },
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

    permute_params{ CASE_PERMUTE_BF16_0, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_1, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_2, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_3, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_4, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_5, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_6, 2, 8 },

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

    permute_params{ CASE_PERMUTE_BF16_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_3D_3, 2, 8 },
    permute_params{ CASE_PERMUTE_BF16_3D_4, 2, 8 },

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
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_0, 2, 8 },
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_1, 2, 8 },
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

    permute_params{ CASE_PERMUTE_BF16_0, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_1, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_2, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_3, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_4, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_5, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_6, 2, 7 },

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

    permute_params{ CASE_PERMUTE_BF16_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_3D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_BF16_3D_4, 2, 7 },

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
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_0, 2, 7 },
    permute_params{ CASE_PERMUTE_XY_SWAP_BF16_1, 2, 7 },
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
#define CASE_PERMUTE_REORDER_BF16_0 { 1, 16, 2, 4 },     { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::bf16, data_types::bf16, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_BF16_1 { 1, 16, 4, 5, 16 }, { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::bf16, data_types::bf16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_BF16_2 { 1, 5, 1, 2, 14 },  { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::bf16, data_types::bf16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_BF16_3 { 1, 16, 2, 4 },     { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::bf16, data_types::bf16,  format::bfyx, format::b_fs_yx_fsv16

// type change
#define CASE_PERMUTE_REORDER_S8_TO_F32_0 { 1, 15, 4, 5 },    { 0, 2, 1, 3 },    { 0, 2, 1, 3 },    data_types::i8, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_S8_TO_F32_1 { 1, 2, 15, 4, 5 }, { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_TO_F16_0 { 1, 5, 1, 2, 14 }, { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::f32, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_U8_TO_F16_0 { 1, 17, 1, 2, 7 },  { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::u8, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_TO_BF16_0 { 1, 5, 1, 2, 14 }, { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::f32, data_types::bf16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_U8_TO_BF16_0 { 1, 17, 1, 2, 7 },  { 0, 2, 1, 3, 4 }, { 0, 1, 2, 3, 4 }, data_types::u8, data_types::bf16, format::b_fs_zyx_fsv16, format::bfzyx

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

// bf16 variants of permute opt reorder cases
#define CASE_PERMUTE_REORDER_TILED_BF16_5 { 1, 48, 3, 256 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2 },  data_types::bf16, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_BF16_6 { 1, 48, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 5, 1, 4, 3, 2 },  data_types::bf16, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx
#define CASE_PERMUTE_REORDER_TILED_BF16_7 { 1, 48, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::bf16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_BF16_8 { 1, 28, 2, 2, 3, 256 }, { 0, 2, 3, 4, 5, 1 }, { 0, 3, 1, 2 },  data_types::bf16, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_BF16_9 { 1, 24, 2, 3, 256 }, { 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 },  data_types::bf16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_BF16_10 { 1, 35, 3, 253 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2 },  data_types::bf16, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_BF16_11 { 1, 32, 3, 253 }, { 0, 2, 3, 1 }, { 0, 5, 1, 4, 2, 3 },  data_types::bf16, data_types::f32, format::bfyx, format::bfwzyx
#define CASE_PERMUTE_REORDER_TILED_BF16_12 { 1, 768, 32, 32 }, { 0, 2, 3, 1 }, { 0, 4, 1, 3, 2},  data_types::bf16, data_types::f32, format::bfyx, format::bfzyx

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
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_1, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_2, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_1, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_TO_F16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_U8_TO_F16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_TO_BF16_0, 4, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_U8_TO_BF16_0, 4, 4 },
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
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_5, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_6, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_7, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_8, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_9, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_10, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_11, 3, 4 },
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
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_3, 3, 4 },
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
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_0, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_1, 4, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_BF16_2, 4, 5 },
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
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_5, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_6, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_7, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_8, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_9, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_10, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_11, 3, 5 },
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
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_7, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_8, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_9, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_10, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_11, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_BF16_12, 3, 5 },
}));

// -----------------------------------------------------------------------------
// Regression coverage for fold_higher_rank_fused_peer() (kernel_selector_helper.h):
// Permute [1,2,8,6,10] bfzyx -> layout optimizer flattens to [1,2,48,10] bfyx for a
// downstream Reshape -> Add fused in, peer stays 5D -> kernel misreads it (MSE ~12-20 vs CPU).
// Uses the real GPU compile/layout/fusion pipeline; GPU output is compared against CPU.
// Requires building and installing the GPU plugin.

namespace {
// can_fuse_reorder_to_prev -> fused_peers_can_fold_to_layout -> fold_higher_rank_fused_peer only
// depends on shape/format/padding, not values, so any fixed-seed weights work. Shared by all models
// in this file (5D and 6D); sized for the largest consumer, smaller ones slice a prefix.
const std::vector<float> kFusedPeerFoldWeights1 = [] {
    tests::random_generator rg;
    rg.set_seed("fused_peer_fold_weights_1");
    return rg.generate_random_1d<float>(96, -2, 2);
}();

const std::vector<float> kFusedPeerFoldWeights2 = [] {
    tests::random_generator rg;
    rg.set_seed("fused_peer_fold_weights_2");
    return rg.generate_random_1d<float>(96, -2, 2);
}();

// Compiles the given model on device, runs one inference with (in1, in2), and returns the f16 output
// flattened to float alongside the compiled model.
std::pair<std::vector<float>, ov::CompiledModel> compile_and_infer(ov::Core& core,
                                                                    const std::string& device,
                                                                    const ov::AnyMap& cfg,
                                                                    const std::shared_ptr<ov::Model>& model,
                                                                    const ov::Tensor& in1,
                                                                    const ov::Tensor& in2) {
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

// Scans a compiled model's runtime graph for the (at most one) node whose ORIGINAL_NAMES rt-info
// contains every string in must_contain and none of must_not_contain. Used by every test below to
// check whether the fused Add is present and at what rank/impl it executed.
struct node_probe {
    bool found = false;
    int64_t rank = -1;
    std::string impl_type;
};

node_probe probe_node(const std::shared_ptr<const ov::Model>& rt,
                       const std::vector<std::string>& must_contain,
                       const std::vector<std::string>& must_not_contain = {}) {
    node_probe result;
    for (const auto& node : rt->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto it = info.find(ov::exec_model_info::ORIGINAL_NAMES);
        if (it == info.end())
            continue;
        const auto orig = it->second.as<std::string>();
        const bool contains_all = std::all_of(must_contain.begin(), must_contain.end(), [&](const std::string& s) {
            return orig.find(s) != std::string::npos;
        });
        if (!contains_all)
            continue;
        const bool excludes_all =
            std::none_of(must_not_contain.begin(), must_not_contain.end(), [&](const std::string& s) {
                return orig.find(s) != std::string::npos;
            });
        if (!excludes_all)
            continue;
        result.found = true;
        result.rank = node->get_output_partial_shape(0).rank().get_length();
        if (auto impl_it = info.find(ov::exec_model_info::IMPL_TYPE); impl_it != info.end())
            result.impl_type = impl_it->second.as<std::string>();
    }
    return result;
}

double fused_peer_fold_mse(const std::vector<float>& a, const std::vector<float>& b, double& max_ae) {
    double se = 0.0;
    max_ae = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        se += d * d;
        max_ae = std::max(max_ae, std::abs(d));
    }
    return se / static_cast<double>(a.size());
}

// GPU_ALLOW_NEW_SHAPE_INFER is a RELEASE_INTERNAL option (options.inl), so ov::Core rejects it when
// passed through the public compile_model() config -- it can only be set via its OV_GPU_ env var.
// Sets the variable for the lifetime of the guard and restores the previous state on destruction.
class scoped_env_var {
public:
    scoped_env_var(std::string name, const std::string& value) : m_name(std::move(name)) {
        if (const char* prev = std::getenv(m_name.c_str())) {
            m_had_prev = true;
            m_prev = prev;
        }
        set(value);
    }

    ~scoped_env_var() {
        if (m_had_prev)
            set(m_prev);
        else
            unset();
    }

    scoped_env_var(const scoped_env_var&) = delete;
    scoped_env_var& operator=(const scoped_env_var&) = delete;

private:
    void set(const std::string& value) {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), value.c_str());
#else
        ::setenv(m_name.c_str(), value.c_str(), 1);
#endif
    }

    void unset() {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), "");
#else
        ::unsetenv(m_name.c_str());
#endif
    }

    std::string m_name;
    std::string m_prev;
    bool m_had_prev = false;
};

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

// Host [1,2,8,6,10] bfzyx -> flattened to [1,2,48,10] bfyx (z=8,y=6 -> 48). Peer [1,pf,pz,py,px] stays
// 5D unless fold_higher_rank_fused_peer() can fold it to the host shape; otherwise rank is preserved.
struct collapse_mask_case {
    int64_t pf, pz, py, px;
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

    auto t1 = make_branch(in1, 2, 8, 6, 10, kFusedPeerFoldWeights1);  // host [1,2,8,6,10]
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, c.pf, c.pz, c.py, c.px, kFusedPeerFoldWeights2);  // peer [1,pf,pz,py,px]
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
                                                                   const ov::AnyMap& cfg,
                                                                   const collapse_mask_case& c,
                                                                   const ov::Tensor& in1,
                                                                   const ov::Tensor& in2) {
    return compile_and_infer(core, device, cfg, build_collapse_mask_model(c), in1, in2);
}

void fill_collapse_mask_inputs(const collapse_mask_case& c, ov::Tensor& in1, ov::Tensor& in2) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto rnd1 = rg.generate_random_1d<ov::float16>(in1.get_size(), -2, 2);
    std::copy(rnd1.begin(), rnd1.end(), in1.data<ov::float16>());
    auto rnd2 = rg.generate_random_1d<ov::float16>(in2.get_size(), -2, 2);
    std::copy(rnd2.begin(), rnd2.end(), in2.data<ov::float16>());
}

class permute_fused_collapse_broadcast_matrix : public ::testing::TestWithParam<collapse_mask_case> {};

}  // namespace

TEST_P(permute_fused_collapse_broadcast_matrix, compiles_finite_and_matches_cpu) {
    const auto c = GetParam();
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, static_cast<size_t>(c.pf), static_cast<size_t>(c.py * c.px), 12});
    fill_collapse_mask_inputs(c, in1, in2);

    auto [cpu_vals, cpu_compiled] = run_collapse_mask(core, "CPU", {}, c, in1, in2);

    std::vector<float> gpu_vals;
    ov::CompiledModel gpu_compiled;
    ASSERT_NO_THROW({
        auto res = run_collapse_mask(core, "GPU", {}, c, in1, in2);
        gpu_vals = res.first;
        gpu_compiled = res.second;
    }) << "Default GPU compilation must not fail for mask "
       << c.label;

    // Assert the intended runtime state: Add fused into Transpose_target, and the host permute rank
    // matches the fix's decision (5D preserved for the inner-spatial masks, 4D flattened otherwise).
    // Also require permute_ref (the fused-eltwise-rank-mismatch defect only manifests via that impl) and,
    // when the host is expected to flatten to 4D, that the peer itself is still seen at its native 5D --
    // confirming the rank-mismatch condition under test is actually present rather than avoided upstream.
    auto rt = gpu_compiled.get_runtime_model();
    auto target = probe_node(rt, {"Transpose_target", "Add_target"});
    ASSERT_TRUE(target.found) << "Add_target not fused into Transpose_target for mask " << c.label;
    EXPECT_TRUE(target.impl_type.find("permute_ref") != std::string::npos)
        << "Target permute did not select permute_ref for mask " << c.label;
    if (c.expect_rank_preserved) {
        EXPECT_EQ(target.rank, 5) << "Inner-spatial mask " << c.label << " must keep the fused permute host at 5D (rank preserved).";
    } else {
        EXPECT_EQ(target.rank, 4) << "Representable mask " << c.label << " should keep the flattened 4D fused host.";
        auto peer = probe_node(rt, {"Transpose_peer"}, {"Add_target"});
        EXPECT_EQ(peer.rank, 5) << "Fused peer dependency was not 5D for mask " << c.label << "; rank-mismatch condition absent.";
    }

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    for (float v : gpu_vals)
        ASSERT_TRUE(std::isfinite(v)) << "Non-finite GPU output for mask " << c.label;
    double max_ae = 0.0;
    double mse = fused_peer_fold_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU vs CPU mismatch for mask " << c.label << " MSE=" << mse << " MaxAbsErr=" << max_ae;
    // FP16 tolerance: correct path agrees with CPU to ~1e-3 MSE; the original defect produced MSE ~12-20.
}

// peer f,z,y,x extents (1 = broadcast), host [1,2,8,6,10]:
//   equal_total              [2,8,6,10] -> folds (original defect shape)
//   feature_broadcast        [1,8,6,10] -> folds (NumPy broadcast, not equal-total)
//   inner_y_broadcast        [2,8,1,10] -> rank preserved
//   inner_y_and_x_broadcast  [2,8,1,1]  -> rank preserved
INSTANTIATE_TEST_SUITE_P(collapse_broadcast_matrix,
                         permute_fused_collapse_broadcast_matrix,
                         ::testing::Values(
                             // host [1,2,8,6,10]; peer f,z,y,x extents (1 == broadcast).
                             collapse_mask_case{2, 8, 6, 10, false, "equal_total"},
                             collapse_mask_case{1, 8, 6, 10, false, "feature_broadcast"},
                             collapse_mask_case{2, 8, 1, 10, true, "inner_y_broadcast"},
                             collapse_mask_case{2, 8, 1, 1, true, "inner_y_and_x_broadcast"}),
                         [](const ::testing::TestParamInfo<collapse_mask_case>& info) {
                             return std::string(info.param.label);
                         });

// New shape-infer: host stays 5D bfzyx -> peer and host are already rank-consistent. Must remain
// correct independent of the flattening path exercised above.
TEST(permute_fused_eltwise_rank_mismatch, new_shape_infer_5d_peer_and_5d_host) {
    ov::Core core;
    if (!discover_gpu_and_cpu(core)) {
        GTEST_SKIP() << "Requires both GPU and CPU plugins discoverable via ov::Core.";
    }

    const collapse_mask_case c{2, 8, 6, 10, false, "equal_total"};
    ov::Tensor in1(ov::element::f16, ov::Shape{1, 2, 60, 12});
    ov::Tensor in2(ov::element::f16, ov::Shape{1, 2, 60, 12});
    fill_collapse_mask_inputs(c, in1, in2);

    auto [cpu_vals, cpu_compiled] = run_collapse_mask(core, "CPU", {}, c, in1, in2);

    std::vector<float> gpu_vals;
    ov::CompiledModel gpu_compiled;
    {
        // New shape-infer is RELEASE_INTERNAL: only settable via env var, not the public config.
        scoped_env_var new_shape_infer_env("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
        ov::Core new_shape_infer_core;
        auto res = run_collapse_mask(new_shape_infer_core, "GPU", {}, c, in1, in2);
        gpu_vals = res.first;
        gpu_compiled = res.second;
    }

    // Fused target permute output stays 5D (rank-consistent with the peer).
    auto rt = gpu_compiled.get_runtime_model();
    auto target = probe_node(rt, {"Transpose_target", "Add_target"});
    ASSERT_TRUE(target.found) << "Add_target was not fused into Transpose_target under new shape-infer.";
    ASSERT_EQ(target.rank, 5) << "Under new shape-infer the fused permute host output should remain 5D.";

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    double max_ae = 0.0;
    double mse = fused_peer_fold_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU (new shape-infer) fused-permute output diverges from CPU reference. "
                            "MSE=" << mse << " MaxAbsErr=" << max_ae;
}

// -----------------------------------------------------------------------------
// 6D-to-4D matrix: same as the 5D matrix above, but folds three spatial axes into
// one (fold_count=2) instead of one. Two cases suffice: one folds, one preserves
// rank. Decision logic itself is unit-tested in canonicalize_fused_shapes_test.cpp.
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

    auto t1 = make_branch(in1, 2, X, 4, 3, 5, kFusedPeerFoldWeights1);  // host [1,2,X,4,3,5]
    t1->set_friendly_name("Transpose_target");
    auto t2 = make_branch(in2, c.pf, c.px, c.pw, c.pz, c.py, kFusedPeerFoldWeights2);
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
    return compile_and_infer(core, device, {}, build_collapse6d_model(c), in1, in2);
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
    auto target = probe_node(rt, {"Transpose_target", "Add_target"});
    ASSERT_TRUE(target.found) << "Add_target not fused into Transpose_target for 6D mask " << c.label;
    if (c.expect_rank_preserved) {
        EXPECT_EQ(target.rank, 6) << "Inner-spatial 6D mask " << c.label << " must preserve 6D host rank.";
    } else {
        EXPECT_EQ(target.rank, 4) << "Representable 6D mask " << c.label << " should keep the 4D host.";
    }

    ASSERT_EQ(gpu_vals.size(), cpu_vals.size());
    for (float v : gpu_vals)
        ASSERT_TRUE(std::isfinite(v)) << "Non-finite GPU output for 6D mask " << c.label;
    double max_ae = 0.0;
    double mse = fused_peer_fold_mse(gpu_vals, cpu_vals, max_ae);
    EXPECT_LT(mse, 1e-2) << "GPU vs CPU mismatch for 6D mask " << c.label << " MSE=" << mse << " MaxAbsErr=" << max_ae;
}

INSTANTIATE_TEST_SUITE_P(collapse6d_matrix,
                         permute_fused_collapse6d_matrix,
                         ::testing::Values(
                             // host [1,2,2,4,3,5]; peer f,x,w,z,y (1 = broadcast):
                             collapse6d_case{2, 2, 4, 3, 5, false, "equal_total_6d"},   // folds
                             collapse6d_case{2, 2, 4, 1, 5, true, "z_broadcast_6d"}),   // preserved
                         [](const ::testing::TestParamInfo<collapse6d_case>& info) {
                             return std::string(info.param.label);
                         });
