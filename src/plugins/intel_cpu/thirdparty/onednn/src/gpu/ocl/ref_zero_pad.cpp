/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include "gpu/ocl/ref_zero_pad.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_zero_pad_t::execute_ref(const exec_ctx_t &ctx) const {
    compute::kernel_arg_list_t arg_list;

    const memory_t *memory = ctx.input(DNNL_ARG_SRC);
    memory_storage_t *mem_storage = memory->memory_storage();
    memory_desc_wrapper mdw(memory->md());

    const int ndims = mdw.ndims();
    const auto &dims = mdw.dims();
    const auto &pdims = mdw.padded_dims();
    const blocking_desc_t blocking_desc = mdw.blocking_desc();
    const ptrdiff_t nelems = (ptrdiff_t)mdw.nelems(true);
    const compute::compute_engine_t *engine
            = utils::downcast<compute::compute_engine_t *>(
                    ctx.stream()->engine());
    const compute::device_info_t *device = engine->device_info();
    const unsigned int hw_threads = device->hw_threads();

    // Setup Initial parameters used in opencl kernel computation
    dims_t blk_size;
    for (int i = 0; i < ndims; i++) {
        blk_size[i] = 1;
    }

    cl_ulong step_nelems = 1;
    for (int i = 0; i < blocking_desc.inner_nblks; i++) {
        step_nelems *= blocking_desc.inner_blks[i];
        blk_size[blocking_desc.inner_idxs[i]] *= blocking_desc.inner_blks[i];
    }

    // This constant needs to be the same as DEFAULT_NELEMS_BLOCK in
    // ref_zero_pad.cl
    const int default_nelems_block = 8;

    // This divisibility condition cannot be changed without some modifications
    // to use of DEFAULT_NELEMS_BLOCK in ref_zero_pad.cl
    size_t nelems_block = 1;
    while (nelems_block < default_nelems_block
            && step_nelems % (nelems_block * 2) == 0)
        nelems_block *= 2;

    arg_list.set(0, *mem_storage);
    arg_list.set(1, mdw.data_type_size());
    arg_list.set(2, step_nelems);
    arg_list.set(3, nelems_block);

    for (int i = 0; i < ndims; i++) {
        if (dims[i] == pdims[i]) continue;
        cl_ulong stride = 1;
        cl_ulong step_count = 1;

        step_count = blocking_desc.strides[i] / step_nelems;
        stride = blocking_desc.strides[i] * (pdims[i] / blk_size[i]);
        size_t npsteps = (nelems / stride) * step_count;

        // Balance work unit size with parallelism
        cl_ulong step_block = 1;
        if (!engine->is_xe_hp() && !engine->is_xe_hpg()) {
            while (step_nelems / nelems_block * step_block < 4 * 1024
                    && step_count % (step_block * 2) == 0
                    && npsteps / step_block > 2 * hw_threads) {
                step_block *= 2;
            }
        }
        dim_t tail_start = dims[i] % blk_size[i];
        dims_t pos;
        for (int j = 0; j < ndims; j++) {
            pos[j] = 0;
        }

        zero_pad_mask_t bit_mask;
        zero_pad_mask_t lookup_mask;
        for (unsigned int j = 0; j < ZERO_PAD_MASK_SIZE; j++)
            bit_mask.mask[j] = 0;

        bool is_done = false;
        bool use_lookup_mask = true;
        size_t mask_count = 0;
        while (!is_done) {
            size_t idx = mdw.off_v(pos, true);
            bool is_valid = pos[i] >= tail_start;
            size_t mask_idx = idx / ZERO_PAD_MASK_DT_BITS;
            size_t mask_bit = idx % ZERO_PAD_MASK_DT_BITS;
            bit_mask.mask[mask_idx] |= (is_valid ? (1 << mask_bit) : 0);
            if (is_valid && use_lookup_mask) {
                if (mask_count < ZERO_PAD_MASK_SIZE
                        && idx <= std::numeric_limits<
                                   ZERO_PAD_MASK_DATA_TYPE>::max()) {
                    lookup_mask.mask[mask_count] = (ZERO_PAD_MASK_DATA_TYPE)idx;
                    mask_count++;
                } else {
                    use_lookup_mask = false;
                }
            }

            //Increment position in the block
            is_done = true;
            for (int j = 0; j < ndims; j++) {
                if (blk_size[j] - 1 == pos[j]) continue;
                is_done = false;
                pos[j] = pos[j] + 1;
                for (int k = j - 1; k >= 0; k--)
                    pos[k] = 0;
                break;
            }
        }

        size_t mode = ZERO_PAD_BIT_MODE;
        size_t gws0 = nelems_block;
        zero_pad_mask_t *mask_in = &bit_mask;
        if (use_lookup_mask) {
            mode = ZERO_PAD_LOOKUP_MODE;
            gws0 = mask_count;
            mask_in = &lookup_mask;
        }

        arg_list.set(4, step_block);
        arg_list.set(5, step_count);
        arg_list.set(6, stride);
        arg_list.set(7, *mask_in);
        arg_list.set(8, mode);

        const size_t gws[3]
                = {gws0, step_count / step_block, npsteps / step_count};
        const compute::nd_range_t nd_range = compute::nd_range_t(3, gws);
        status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
        if (status != status::success) return status;
    }
    return status::success;
}

status_t ref_zero_pad_t::execute_subg_16(const exec_ctx_t &ctx,
        const memory_desc_wrapper &mdw,
        const blocking_desc_t &blocking_desc) const {

    const memory_t *memory = ctx.input(DNNL_ARG_SRC);
    const memory_storage_t *mem_storage = memory->memory_storage();

    const int ndims = mdw.ndims();
    const auto &dims = mdw.dims();
    const auto &pdims = mdw.padded_dims();
    const auto mem_total_size = mdw.size();

    const auto most_inner_nblk = blocking_desc.inner_nblks - 1;

    const unsigned mem_dt_size = static_cast<unsigned>(mdw.data_type_size());

    const cl_ulong most_inner_block_size
            = mem_dt_size * blocking_desc.inner_blks[most_inner_nblk];

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *mem_storage);
    arg_list.set(1, mem_dt_size);
    arg_list.set(3, most_inner_block_size);

    int arg_idx = 0;
    size_t gws2 = 1;
    const size_t lws[3] = {16, 1, 1};

    for (int j = 0; j < MAX_NDIMS; ++j) {
        if (j != blocking_desc.inner_idxs[most_inner_nblk]
                && j != blocking_desc.inner_idxs[most_inner_nblk - 1]) {
            assert(arg_idx < 4);
            if (j < ndims) {
                arg_list.set(5 + arg_idx,
                        mem_dt_size * (cl_ulong)blocking_desc.strides[j]);
                arg_list.set(9 + arg_idx++, (unsigned)dims[j]);
                gws2 *= dims[j];
            } else {
                arg_list.set(5 + arg_idx, cl_ulong(0));
                arg_list.set(9 + arg_idx++, unsigned(1));
            }
        }
    }

    status_t status;
    dims_t coordinates;

    if (pdims[blocking_desc.inner_idxs[most_inner_nblk]]
            != dims[blocking_desc.inner_idxs[most_inner_nblk]]) {
        for (int j = 0; j < ndims; ++j) {
            coordinates[j] = 0;
        }
        coordinates[blocking_desc.inner_idxs[most_inner_nblk]]
                = dims[blocking_desc.inner_idxs[most_inner_nblk]];
        const cl_ulong most_inner_block_base_offset
                = mem_dt_size * mdw.off_v(coordinates, true);

        const cl_ulong s2most_inner_block_stride = mem_dt_size
                * blocking_desc.strides[blocking_desc.inner_idxs[most_inner_nblk
                        - 1]];
        const unsigned most_inner_block_write_multiplier
                = (pdims[blocking_desc.inner_idxs[most_inner_nblk]]
                          - dims[blocking_desc.inner_idxs[most_inner_nblk]])
                / 16;

        arg_list.set(2, most_inner_block_base_offset);
        arg_list.set(4, s2most_inner_block_stride);
        arg_list.set(13, most_inner_block_write_multiplier);

        const size_t gws0 = 16
                * nstl::min<dnnl_dim_t>(
                        dims[blocking_desc.inner_idxs[most_inner_nblk - 1]],
                        blocking_desc.inner_blks[most_inner_nblk - 1]);
        const size_t gws1 = nstl::max<dnnl_dim_t>(
                dims[blocking_desc.inner_idxs[most_inner_nblk - 1]]
                        / blocking_desc.inner_blks[most_inner_nblk - 1],
                1);
        const size_t gws[3] = {gws0, gws1, gws2};
        const compute::nd_range_t zp_nd_range
                = compute::nd_range_t(3, gws, lws);

        status = parallel_for(ctx, zp_nd_range, kernel_subg16_, arg_list);
        CHECK(status);

        if (dims[blocking_desc.inner_idxs[most_inner_nblk - 1]]
                        != pdims[blocking_desc.inner_idxs[most_inner_nblk - 1]]
                && s2most_inner_block_stride != mem_total_size) {
            const cl_ulong base_offset_b2 = most_inner_block_base_offset
                    + s2most_inner_block_stride * gws1;
            arg_list.set(2, base_offset_b2);

            const size_t gws_10 = 16
                    * (dims[blocking_desc.inner_idxs[most_inner_nblk - 1]]
                            % blocking_desc.inner_blks[most_inner_nblk - 1]);
            const size_t gws_1[3] = {gws_10, 1, gws2};
            const compute::nd_range_t zp_nd_range1
                    = compute::nd_range_t(3, gws_1, lws);
            status = parallel_for(ctx, zp_nd_range1, kernel_subg16_, arg_list);
            CHECK(status);
        }
    }

    for (int j = 0; j < ndims; ++j) {
        coordinates[j] = 0;
    }
    coordinates[blocking_desc.inner_idxs[most_inner_nblk - 1]]
            = dims[blocking_desc.inner_idxs[most_inner_nblk - 1]];
    const cl_ulong s2most_inner_block_base_offset
            = mem_dt_size * mdw.off_v(coordinates, true);

    const cl_ulong most_inner_block_offset = mem_dt_size
            * blocking_desc.strides[blocking_desc.inner_idxs[most_inner_nblk]];

    const unsigned most_inner_block_write_multiplier = nstl::max<dnnl_dim_t>(
            blocking_desc.inner_blks[most_inner_nblk] / 16, 1);

    arg_list.set(2, s2most_inner_block_base_offset);
    arg_list.set(4, most_inner_block_offset);
    arg_list.set(13, most_inner_block_write_multiplier);

    const size_t gws0
            = ((pdims[blocking_desc.inner_idxs[most_inner_nblk - 1]]
                       - dims[blocking_desc.inner_idxs[most_inner_nblk - 1]])
                      * blocking_desc.inner_blks[most_inner_nblk])
            / most_inner_block_write_multiplier;
    const size_t gws1 = nstl::max<dnnl_dim_t>(
            pdims[blocking_desc.inner_idxs[most_inner_nblk]]
                    / blocking_desc.inner_blks[most_inner_nblk],
            1);
    const size_t gws[3] = {gws0, gws1, gws2};

    const compute::nd_range_t zp_nd_range = compute::nd_range_t(3, gws, lws);
    status = parallel_for(ctx, zp_nd_range, kernel_subg16_, arg_list);

    return status;
}

status_t ref_zero_pad_t::execute_subg_16_mask_and_clear_dt_1B(
        const exec_ctx_t &ctx, const memory_desc_wrapper &mdw,
        const blocking_desc_t &blocking_desc) const {

    const memory_t *memory = ctx.input(DNNL_ARG_SRC);
    const memory_storage_t *mem_storage = memory->memory_storage();

    const compute::compute_engine_t *engine
            = utils::downcast<compute::compute_engine_t *>(
                    ctx.stream()->engine());
    const compute::device_info_t *device = engine->device_info();
    const compute::gpu_arch_t gpu_gen = device->gpu_arch();
    const size_t max_local_ws = static_cast<size_t>(
            gpu_gen == compute::gpu_arch_t::gen9 ? 256 : 512);

    const auto &dims = mdw.dims();
    const auto nelems = mdw.nelems(true);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *mem_storage);

    const unsigned mask
            = dims[blocking_desc.inner_idxs[0]] % blocking_desc.inner_blks[0];
    arg_list.set(1, mask);

    const unsigned block_size = 16 * 8; // SIMD * block_size
    const size_t gws[3] = {static_cast<size_t>(16 * nelems / block_size), 1, 1};
    const size_t lws[3] = {max_local_ws, 1, 1};

    const compute::nd_range_t zp_nd_range = compute::nd_range_t(3, gws, lws);

    return parallel_for(
            ctx, zp_nd_range, kernel_subg16_mask_and_clear_dt_1b_, arg_list);
}

status_t ref_zero_pad_t::execute(const exec_ctx_t &ctx) const {
    const memory_t *memory = ctx.input(DNNL_ARG_SRC);
    const memory_desc_wrapper mdw(memory->md());
    const blocking_desc_t &blocking_desc = mdw.blocking_desc();

    using namespace format_tag;
    if (blocking_desc.inner_nblks == 2
            && mdw.dims()[blocking_desc.inner_idxs[1]] % 16 == 0
            && blocking_desc.inner_blks[1] % 16 == 0) {
        return execute_subg_16(ctx, mdw, blocking_desc);
    } else if (blocking_desc.inner_nblks == 1
            && blocking_desc.inner_blks[0] == 32
            && mdw.dims()[blocking_desc.inner_idxs[0]] < 16
            && (mdw.nelems(true) % 4096) == 0 && mdw.data_type_size() == 1) {
        return execute_subg_16_mask_and_clear_dt_1B(ctx, mdw, blocking_desc);
    } else {
        return execute_ref(ctx);
    }
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
