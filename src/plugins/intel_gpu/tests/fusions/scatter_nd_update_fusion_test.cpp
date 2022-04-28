// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>

#include <cmath>

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
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, get_default_format(p.indices_rank), p.indices_shape };
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

    template<typename T>
    T generate_random_val(int min, int max, int k = 8) {
        static std::default_random_engine generator(random_seed);
        // 1/k is the resolution of the floating point numbers
        std::uniform_int_distribution<int> distribution(k * min, k * max);
        T val = (T)distribution(generator);
        val /= k;

        return val;
    }

    template <typename T>
    std::vector<T> generate_unique_indices(scatter_nd_update_test_params& p) {
        std::set<std::vector<T>> unique_indices;
        std::vector<T> result;
        auto indices_shape = p.indices_shape.sizes(get_default_format(p.indices_rank));
        auto last_indices_dim = indices_shape.back();

        auto count = 1;
        for (size_t i = 0; i < indices_shape.size() - 1; i++)
            count *= indices_shape[i];

        while (unique_indices.size() != count) {
            std::vector<T> indices;
            for (size_t i = 0; i < last_indices_dim; i++)
                indices.push_back(generate_random_val<T>(0, indices_shape[i]));

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
        if (indices_layout.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_unique_indices<float>(p);
            set_values(prim, rnd_vec);
            // std::cout << "get_indices_mem" << std::endl;
            // for(int i = 0; i < rnd_vec.size(); ++i)
            // {
            //     if(i != 0 && i % indices_layout.get_dims().at(p.indices_rank - 1) == 0)
            //         std::cout << std::endl;
            //     std::cout << std::setw(10) << rnd_vec[i];
            // }
            // std::cout << std::endl;
        } else if (indices_layout.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_unique_indices<FLOAT16>(p);
            // std::cout << "get_indices_mem" << std::endl;
            // for(int i = 0; i < rnd_vec.size(); ++i)
            // {
            //     if(i != 0 && i % indices_layout.get_dims().at(p.indices_rank - 1) == 0)
            //         std::cout << std::endl;
            //     std::cout << std::setw(10) << (float)rnd_vec[i];
            // }
            // std::cout << std::endl;
            set_values(prim, rnd_vec);
        } else if (indices_layout.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_unique_indices<int8_t>(p);
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
#define CASE_SCATTER_ND_UPDATE_FP16_6D_5 { 6, 7, 8, 9, 6, 6 }, { 5, 5, 1, 1 }, { 5, 8, 1, 1, 1, 1 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_6 { 6, 7, 8, 9, 6, 6 }, { 5, 6, 1, 1 }, { 5, 1, 1, 1, 1, 1 }, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx

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
#define CASE_SCATTER_ND_UPDATE_FP32_6D_5 { 6, 7, 8, 9, 6, 6 }, { 5, 5, 1, 1 }, { 5, 8, 1, 1, 1, 1 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_6 { 6, 7, 8, 9, 6, 6 }, { 5, 6, 1, 1 }, { 5, 1, 1, 1, 1, 1 }, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx

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

#define CASE_SCATTER_ND_UPDATE_I8_FSV16_4D_1 { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::bfyx

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
        scatter_nd_update("scatter_nd_update_prim", "input", "scatter_nd_update_indices", "scatter_nd_update_updates", p.indices_rank),
        quantize("quantize", "scatter_nd_update_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.input_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_quantize, ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 3 },  // 0
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 3 },  // 6
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_7, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_9, 2, 3 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 3 },  // 13
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 3 },  // 19
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 3 },  // 25
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 3 },  // 30
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 3 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_1, 2, 3 },  // FP16
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_1, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_6, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_7, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_8, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_9, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_1, 2, 3 },  // FP32
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_1, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_6, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_7, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_8, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_9, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_1, 2, 3 },  // FP16
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_1, 2, 3 },  // FP32
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_2, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_3, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_4, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_5, 2, 3 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_6, 2, 3 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_I8_FSV16_4D_1, 2, 3 },
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
        scatter_nd_update("scatter_nd_update_prim", "input", "scatter_nd_update_indices", "scatter_nd_update_updates", p.indices_rank),
        activation("activation", "scatter_nd_update_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        eltwise("eltwise", { "scale", "eltwise_data" }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.input_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_scale_activation_eltwise, ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 5 },  // 0
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 5 },  // 6
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 5 },  // 12
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 5 },

    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 5 },  // 18
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 5 },  // 24
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 5 },  // 29
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 5 },
    scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_1, 2, 5 },  // FP16
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_4D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_1, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_6, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_7, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_8, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_FSV16_5D_9, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_1, 2, 5 },  // FP32
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_4D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_1, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_6, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_7, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_8, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_FSV16_5D_9, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_1, 2, 5 },  // FP16
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_BSV32_FSV16_4D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_1, 2, 5 },  // FP32
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_2, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_3, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_4, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_5, 2, 5 },
    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_BSV32_FSV16_4D_6, 2, 5 },

    // scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_I8_FSV16_4D_1, 2, 5 },
}));
