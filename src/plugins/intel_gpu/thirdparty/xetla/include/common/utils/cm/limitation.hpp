/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/core/cm/common.hpp"
#include "../../../common/utils/cm/tensor_descriptor.hpp"
#else
#include "common/core/cm/common.hpp"
#include "common/utils/cm/tensor_descriptor.hpp"
#endif

#define IN_RANGE(x, l, r) ((x) >= (l) && (x) <= (r))

namespace gpu::xetla {

typedef uint64_t size_t;

namespace core {
template <gpu_arch arch, typename T>
struct general_1d {
    template <uint8_t NElts>
    static inline bool check_restriction(uint64_t offset, uint64_t p = 0) {
        return true;
    }

    template <uint8_t NElts, int N, typename Toffset = uint32_t>
    static inline bool check_restriction(
            xetla_vector<Toffset, N> offsets, uint64_t p = 0) {
        return true;
    }
};

template <gpu_arch arch, typename T>
class block_2d {
public:
    template <bool transpose, bool vnni_transform>
    static inline bool check_load(xetla_tdescriptor tdesc) {
        return true;
    }

    static inline bool check_store(xetla_tdescriptor tdesc) { return true; }

    static inline bool check_tensor(
            uint64_t base, uint32_t width, uint32_t height, uint32_t pitch) {
        return true;
    }
};
} // namespace core

namespace subgroup {
template <gpu_arch arch, typename dtype, typename mem_dtype>
struct check_load {
    template <bool mem_transform, size_t block_size_x>
    struct global_2d {};

    struct global_1d {};

    template <bool mem_transform, size_t block_size_x>
    struct unaligned_2d {};

    template <mem_layout memory_layout, size_t block_size_x, size_t tile_bytes,
            size_t min_bytes, size_t block_bytes, size_t num_channel_x,
            size_t num_channel>
    struct local_scatter {};

    struct local_1d {};
};

template <gpu_arch arch, typename dtype, typename mem_dtype = uint32_t>
struct check_store {
    template <size_t block_size_x>
    struct global_2d {};

    struct global_1d {};

    template <size_t block_size_x>
    struct unaligned_2d {};

    template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct global_atomic {};

    template <size_t tile_bytes, size_t min_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct local_scatter {};

    template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct local_scatter_vnni_col {};

    struct local_1d {};
};
} // namespace subgroup

namespace group {

template <gpu_arch arch>
struct gemm {
    struct default_fpu {
        template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
                typename dtype_mma_b, typename dtype_mma_acc>
        struct check_dtype_default {};

        template <mem_layout mem_layout_a, mem_layout mem_layout_b,
                mem_space mem_space_a, mem_space mem_space_b>
        struct check_memory_default {};

        template <typename dtype_mma, int tile_size_x_a, int tile_size_y_a,
                int block_size_x_a, int block_size_y_a, int tile_size_x_b,
                int tile_size_y_b, int block_size_x_b, int block_size_y_b>
        struct check_tile_size_default {};
    };

    struct default_xmx {
        template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
                typename dtype_mma_b>
        struct check_dtype_default {};

        template <mem_layout mem_layout_a, mem_layout mem_layout_b,
                mem_space mem_space_a, mem_space mem_space_b>
        struct check_memory_default {};

        template <typename dtype_mma, int tile_size_x_a, int tile_size_y_a,
                int block_size_x_a, int block_size_y_a, int tile_size_x_b,
                int tile_size_y_b, int block_size_x_b, int block_size_y_b>
        struct check_tile_size_default {};
    };
};
} // namespace group

namespace kernel {
template <gpu_arch arch, typename T>
class general_1d {
public:
    static inline bool check_alignment(T *base, uint32_t pitch) { return true; }
};
} // namespace kernel

} // namespace gpu::xetla
