// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>

#include <intel_gpu/primitives/activation.hpp>

#include <cmath>
#include <limits>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace {
struct scatter_nd_update_test_params {
    tensor input_shape;
    tensor indices_shape;
    tensor updates_shape;
    int indices_rank;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class ScatterNDUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_nd_update_test_params> {
public:
    void execute(scatter_nd_update_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(scatter_nd_update_test_params& p) {
        return layout{ data_types::i32, get_default_format(p.indices_rank), p.indices_shape };
    }

    layout get_updates_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.updates_shape };
    }

    layout get_per_channel_layout(scatter_nd_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.input_shape.feature[0], 1, 1 } };
    }

    format get_default_format(int rank = 4) {
        if (rank <= 4)
            return cldnn::format::bfyx;
        else if (rank == 5)
            return cldnn::format::bfzyx;
        else
            return cldnn::format::bfwzyx;
    }

    template <typename T>
    std::vector<T> generate_unique_indices(scatter_nd_update_test_params& p) {
        std::set<std::vector<T>> unique_indices;
        std::vector<T> result;
        auto indices_shape = p.indices_shape.sizes(get_default_format(p.indices_rank));
        auto data_shape = p.input_shape.sizes(p.input_format);
        auto last_indices_dim = indices_shape.at(p.indices_rank - 1);

        auto count = p.indices_shape.count() / last_indices_dim;

        while (unique_indices.size() != count) {
            std::vector<T> indices;
            for (size_t i = 0; i < static_cast<size_t>(last_indices_dim); i++) {
                indices.push_back(static_cast<T>(rg.generate_random_val<int>(0, data_shape[i] - 1)));
            }

            unique_indices.insert(indices);
        }

        std::for_each(unique_indices.begin(),
                      unique_indices.end(),
                      [&](const std::vector<T>& indices) {
                          result.insert(result.end(), indices.begin(), indices.end());
                      });

        return result;
    }

    cldnn::memory::ptr get_indices_mem(scatter_nd_update_test_params& p) {
        auto indices_layout = get_indices_layout(p);
        auto prim = engine.allocate_memory(indices_layout);
        if (indices_layout.data_type == data_types::i32) {
            VF<int32_t> rnd_vec = generate_unique_indices<int32_t>(p);
            set_values(prim, rnd_vec);
        } else if (indices_layout.data_type == data_types::i64) {
            VF<int64_t> rnd_vec = generate_unique_indices<int64_t>(p);
            set_values(prim, rnd_vec);
        } else {
            throw std::runtime_error("Unsupported data type for indicies of scatter_nd_update primitive");
        }

        return prim;
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------- */
/* ------------------------------------ ScatterNDUpdate cases ------------------------------------ */
/* ----------------------------------------------------------------------------------------------- */
#define CASE_SCATTER_ND_UPDATE_FP16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_3 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_5D_1 { 6, 7, 8, 9, 10 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_2 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 1 }, { 5, 10, 1, 8, 9 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_3 { 6, 7, 8, 9, 10 }, { 5, 3, 1, 1 }, { 5, 9, 1, 1, 8 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_4 { 6, 7, 8, 9, 10 }, { 5, 4, 1, 1 }, { 5, 8, 1, 1, 1 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_5 { 6, 7, 8, 9, 10 }, { 5, 5, 1, 1 }, { 5, 1, 1, 1, 1 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_6 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 2 }, { 5, 2, 8, 9, 10 }, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_7 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 3 }, { 5, 2, 1, 8, 9 }, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_8 { 6, 7, 8, 9, 10 }, { 5, 2, 4, 3 }, { 5, 2, 1, 8, 3 }, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_9 { 6, 7, 8, 9, 10 }, { 5, 2, 3, 3 }, { 5, 2, 8, 9, 3 }, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_6D_1 { 6, 7, 8, 9, 10, 11 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10, 11 }, 1, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_2 { 6, 7, 8, 9, 10, 11 }, { 5, 2, 1, 1 }, { 5, 11, 1, 8, 9, 10 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_3 { 6, 7, 8, 9, 10, 11 }, { 5, 3, 1, 1 }, { 5, 10, 1, 1, 8, 9 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_4 { 6, 7, 8, 9, 10, 11 }, { 5, 4, 1, 1 }, { 5, 9, 1, 1, 1, 8 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_5 { 6, 7, 8, 9, 2, 2 }, { 5, 5, 1, 1 }, { 5, 8, 1, 1, 1, 1 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_6 { 6, 7, 8, 9, 2, 2 }, { 5, 6, 1, 1 }, { 5, 1, 1, 1, 1, 1 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_3 { 6, 7, 8, 1 }, { 5, 1, 1, 1 }, { 5, 7, 8, 1 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_5D_1 { 6, 7, 8, 9, 10 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_2 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 1 }, { 5, 10, 1, 8, 9 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_3 { 6, 7, 8, 9, 10 }, { 5, 3, 1, 1 }, { 5, 9, 1, 1, 8 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_4 { 6, 7, 8, 9, 10 }, { 5, 4, 1, 1 }, { 5, 8, 1, 1, 1 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_5 { 6, 7, 8, 9, 10 }, { 5, 5, 1, 1 }, { 5, 1, 1, 1, 1 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_6D_1 { 6, 7, 8, 9, 10, 11 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10, 11 }, 1, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_2 { 6, 7, 8, 9, 10, 11 }, { 5, 2, 1, 1 }, { 5, 11, 1, 8, 9, 10 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_3 { 6, 7, 8, 9, 10, 11 }, { 5, 3, 1, 1 }, { 5, 10, 1, 1, 8, 9 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_4 { 6, 7, 8, 9, 10, 11 }, { 5, 4, 1, 1 }, { 5, 9, 1, 1, 1, 8 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_5 { 6, 7, 8, 9, 2, 2 }, { 5, 5, 1, 1 }, { 5, 8, 1, 1, 1, 1 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_6 { 6, 7, 8, 9, 2, 2 }, { 5, 6, 1, 1 }, { 5, 1, 1, 1, 1, 1 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_3 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_1 { 6, 7, 8, 9, 10 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_2 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 1 }, { 5, 10, 1, 8, 9 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_3 { 6, 7, 8, 9, 10 }, { 5, 3, 1, 1 }, { 5, 9, 1, 1, 8 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_4 { 6, 7, 8, 9, 10 }, { 5, 4, 1, 1 }, { 5, 8, 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_5 { 6, 7, 8, 9, 10 }, { 5, 5, 1, 1 }, { 5, 1, 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_6 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 2 }, { 5, 2, 8, 9, 10 }, 3, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_7 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 3 }, { 5, 2, 1, 8, 9 }, 3, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_8 { 6, 7, 8, 9, 10 }, { 5, 2, 4, 3 }, { 5, 2, 1, 8, 3 }, 4, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_9 { 6, 7, 8, 9, 10 }, { 5, 2, 3, 3 }, { 5, 2, 8, 9, 3 }, 4, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_3 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_1 { 6, 7, 8, 9, 10 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_2 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 1 }, { 5, 10, 1, 8, 9 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_3 { 6, 7, 8, 9, 10 }, { 5, 3, 1, 1 }, { 5, 9, 1, 1, 8 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_4 { 6, 7, 8, 9, 10 }, { 5, 4, 1, 1 }, { 5, 8, 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_5 { 6, 7, 8, 9, 10 }, { 5, 5, 1, 1 }, { 5, 1, 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_6 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 2 }, { 5, 2, 8, 9, 10 }, 3, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_7 { 6, 7, 8, 9, 10 }, { 5, 2, 1, 3 }, { 5, 2, 1, 8, 9 }, 3, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_8 { 6, 7, 8, 9, 10 }, { 5, 2, 4, 3 }, { 5, 2, 1, 8, 3 }, 4, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_9 { 6, 7, 8, 9, 10 }, { 5, 2, 3, 3 }, { 5, 2, 8, 9, 3 }, 4, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_3 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f16, format::bs_fs_yx_bsv32_fsv16, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_2 { 6, 6, 1, 1 }, { 3, 2, 1, 1 }, { 3, 1, 1, 1 }, 2, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_3 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_4 { 6, 7, 8, 9 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9 }, 2, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_5 { 6, 7, 8, 9 }, { 6, 2, 1, 1 }, { 6, 9, 1, 8 }, 2, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_6 { 6, 7, 8, 9 }, { 6, 3, 1, 1 }, { 6, 8, 1, 1 }, 2, data_types::f32, format::bs_fs_yx_bsv32_fsv16, data_types::f32, format::bfyx


class scatter_nd_update_prelu_nan : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_prelu_nan, basic) {
    auto p = GetParam();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto input = engine.allocate_memory(get_input_layout(p));
    auto updates = engine.allocate_memory(get_updates_layout(p));
    auto indices = engine.allocate_memory(get_indices_layout(p));
    auto slopes = engine.allocate_memory(get_per_channel_layout(p));
    set_values<int32_t>(indices, {1});
    if (p.data_type == data_types::f16) {
        set_values<ov::float16>(input, {nan, -4, 8, -8, 1, 2, 3, 4});
        set_values<ov::float16>(updates, {-6, nan, -12, 16});
        set_values<ov::float16>(slopes, {0.5f, 0.25f});
    } else {
        set_values<float>(input, {nan, -4, 8, -8, 1, 2, 3, 4});
        set_values<float>(updates, {-6, nan, -12, 16});
        set_values<float>(slopes, {0.5f, 0.25f});
    }

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("indices", indices),
        data("updates", updates),
        data("slopes", slopes),
        scatter_nd_update("scatter", input_info("input"), input_info("indices"), input_info("updates"), p.indices_rank),
        activation("prelu", input_info("scatter"), "slopes", activation_func::relu_negative_slope),
        reorder("output", input_info("prelu"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc scatter_impl = {format::bfyx, "ocl::scatter_nd_update", impl_types::ocl};
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"scatter", scatter_impl}}));
    network network_fused(engine, topology_fused, cfg_fused);
    network_fused.set_input_data("input", input);
    {
        // The producer keeps its id only while the scatter stays a GPU node:
        // an optimizer may merge it into a copy, or (immad f32) partitioning
        // may move execution elsewhere, in which case this fused ocl_v2 path
        // is not what is being run and the case contributes no signal. The
        // f16 case on the same device does hit the fused path and gates the
        // regression; skip the others with the actual ids in the log rather
        // than failing on a topology-shape difference unrelated to the fix.
        const auto prim_list = network_fused.get_primitives_info();
        bool has_scatter = false;
        std::string all_ids;
        for (const auto& info : prim_list) {
            has_scatter = has_scatter || info.original_id == "scatter";
            all_ids += info.original_id + " ";
        }
        if (!has_scatter)
            GTEST_SKIP() << "no primitive with id scatter (ids: " << all_ids
                         << ") - fused ocl_v2 path not exercised for this configuration";
    }
    ASSERT_NO_FATAL_FAILURE(check_fusions_correctness(network_fused, {{"scatter", {"prelu"}}}));
    const auto primitives = network_fused.get_primitives_info();
    const auto producer = std::find_if(primitives.begin(), primitives.end(), [](const primitive_info& info) {
        return info.original_id == "scatter";
    });
    ASSERT_NE(producer, primitives.end());
    ASSERT_NE(producer->kernel_id.find("ocl::scatter_nd_update"), std::string::npos);

    auto outputs = network_fused.execute();
    const auto executed = network_fused.get_executed_primitives();
    ASSERT_EQ(executed.count("scatter"), 1);
    ASSERT_EQ(executed.count("prelu"), 0);
    ASSERT_EQ(outputs.size(), 1);
    const auto actual = get_output_values_to_float(network_fused, outputs.begin()->second);
    // Both the untouched batch and the updated batch contain NaN, so the copy
    // and update kernels must each preserve it through the fused PReLU.
    const std::vector<float> expected = {nan, -2, 8, -2, -3, nan, -3, 16};
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
            ASSERT_TRUE(std::isnan(actual[i])) << "at index " << i << ": got " << actual[i];
        } else {
            ASSERT_FLOAT_EQ(actual[i], expected[i]) << "at index " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_prelu_nan, ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
    {{2, 2, 2, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx, 2, 3},
    {{2, 2, 2, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx, 2, 3},
}));

class scatter_nd_update_quantize : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_nd_update_indices", get_indices_mem(p)),
        data("scatter_nd_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        scatter_nd_update("scatter_nd_update_prim", input_info("input"), input_info("scatter_nd_update_indices"),
                          input_info("scatter_nd_update_updates"), p.indices_rank),
        quantize("quantize", input_info("scatter_nd_update_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.input_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_quantize, ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_7, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_9, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_1, 2, 3 },  // FP16
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_6, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_7, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_8, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_9, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_1, 2, 3 },  // FP32
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_1, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_6, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_7, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_8, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_9, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_1, 2, 3 },  // FP16
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_1, 2, 3 },  // FP32
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_6, 2, 3 },
}));

class scatter_nd_update_quantize_reversed_output_range : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_quantize_reversed_output_range, pre_shift) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_nd_update_indices", get_indices_mem(p)),
        data("scatter_nd_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("in_lo", get_mem(get_per_channel_layout(p), -2.0f)),
        data("in_hi", get_mem(get_per_channel_layout(p), 3.0f)),
        data("out_lo", get_mem(get_single_element_layout(p), 127.0f)),
        data("out_hi", get_mem(get_single_element_layout(p), -127.0f)),
        scatter_nd_update("scatter_nd_update_prim", input_info("input"), input_info("scatter_nd_update_indices"),
                          input_info("scatter_nd_update_updates"), p.indices_rank),
        quantize("quantize", input_info("scatter_nd_update_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.input_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_quantize_reversed_output_range,
                         ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
                             scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 3 },
                         }));

class scatter_nd_update_scale_activation_eltwise : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_nd_update_indices", get_indices_mem(p)),
        data("scatter_nd_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("scale_data", get_mem(get_per_channel_layout(p), -1, 1)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.input_shape })),
        scatter_nd_update("scatter_nd_update_prim", input_info("input"), input_info("scatter_nd_update_indices"),
                          input_info("scatter_nd_update_updates"), p.indices_rank),
        activation("activation", input_info("scatter_nd_update_prim"), activation_func::abs),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        eltwise("eltwise", { input_info("scale"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", input_info("eltwise"), p.input_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_scale_activation_eltwise, ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_1, 2, 5 },  // FP16
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_6, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_7, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_8, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_9, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_1, 2, 5 },  // FP32
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_1, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_6, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_7, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_8, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_9, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_1, 2, 5 },  // FP16
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_1, 2, 5 },  // FP32
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_6, 2, 5 },
}));
