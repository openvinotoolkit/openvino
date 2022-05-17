/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <math.h>

#include "common/primitive_exec_types.hpp"

#include "gpu/ocl/gen9_reduction.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

std::pair<int, int> get_n_c_block_sizes(const memory_desc_wrapper &mdw) {
    int n_block_size = 1;
    int c_block_size = 1;
    const blocking_desc_t &blk = mdw.blocking_desc();
    if (blk.inner_nblks > 0) {
        // C must be the last blocked dimension
        assert(blk.inner_idxs[blk.inner_nblks - 1] == 1);
        c_block_size = blk.inner_blks[blk.inner_nblks - 1];
        // if there is NC blocking (N is the blocked dimension before C) use N blocks as well
        if (blk.inner_nblks > 1 && blk.inner_idxs[blk.inner_nblks - 2] == 0) {
            n_block_size = blk.inner_blks[blk.inner_nblks - 2];
        }
    }
    return std::make_pair(n_block_size, c_block_size);
}

std::pair<int, int> get_initial_n_split(const int n, const bool is_n_reduced) {
    int initial_n_chunk_size;
    int initial_n_chunks_num;
    if (is_n_reduced) {
        // Start with such constant and try to adjust that with heuristics
        initial_n_chunk_size = 64;
        while (initial_n_chunk_size > n) {
            initial_n_chunk_size /= 2;
        }
        initial_n_chunks_num
                = ceil(static_cast<float>(n) / initial_n_chunk_size);
        // We don't want to have too many chunks as there would be a lot of work for
        // final reduction. Desired values were selected experimentally.
        int desired_n_chunks = 16;
        constexpr int min_chunk_size = 4;
        if (n / min_chunk_size < desired_n_chunks && n / min_chunk_size >= 1) {
            desired_n_chunks = n / min_chunk_size;
        }
        int desired_chunk_size = 32;
        if (n / desired_n_chunks < desired_chunk_size) {
            desired_chunk_size = n / desired_n_chunks;
        }
        while (initial_n_chunk_size < desired_chunk_size
                && initial_n_chunks_num > desired_n_chunks
                && initial_n_chunk_size * 2 < n) {
            initial_n_chunk_size *= 2;
            initial_n_chunks_num
                    = ceil(static_cast<float>(n) / initial_n_chunk_size);
        }
    } else {
        initial_n_chunks_num = n;
        initial_n_chunk_size = 1;
    }
    return std::make_pair(initial_n_chunk_size, initial_n_chunks_num);
}

status_t gen9_reduction_t::pd_t::init_conf(engine_t *engine) {
    const reduction_pd_t *pd = this;

    const memory_desc_wrapper src_mdw(pd->src_md());
    const memory_desc_wrapper dst_mdw(pd->dst_md());

    const int ndims = src_mdw.ndims();
    const dnnl_dim_t *src_dims = src_mdw.md_->dims;
    const dnnl_dim_t *dst_dims = dst_mdw.md_->dims;
    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    const int num_threads = compute_engine->device_info()->hw_threads();

    conf.alg = pd->desc()->alg_kind;
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.dst_type = dst_mdw.data_type();
    conf.src_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.power = pd->desc()->p;
    conf.eps = pd->desc()->eps;
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);
    conf.finilize_dispatch = compute_engine->create_dispatch();

    auto is_c_blocked_by
            = [](const memory_desc_wrapper &mdw, const int blockSize) {
                  const blocking_desc_t &blk = mdw.blocking_desc();
                  if (blk.inner_nblks == 0) return false;
                  return (blk.inner_idxs[blk.inner_nblks - 1] == 1)
                          && (blk.inner_blks[blk.inner_nblks - 1] == blockSize);
              };

    if (!(is_c_blocked_by(src_mdw, 16) || is_c_blocked_by(src_mdw, 32)))
        return status::unimplemented;

    int src_n_block_size, src_c_block_size;
    int dst_n_block_size, dst_c_block_size;
    std::tie(src_n_block_size, src_c_block_size) = get_n_c_block_sizes(src_mdw);
    std::tie(dst_n_block_size, dst_c_block_size) = get_n_c_block_sizes(dst_mdw);
    if (src_n_block_size != dst_n_block_size
            || src_c_block_size != dst_c_block_size
            || src_mdw.blocking_desc().inner_nblks
                    != dst_mdw.blocking_desc().inner_nblks)
        return status::unimplemented;

    conf.n_block_size = src_n_block_size;
    conf.c_block_size = src_c_block_size;
    if ((conf.n_block_size == 1 && src_mdw.blocking_desc().inner_nblks > 1)
            || src_mdw.blocking_desc().inner_nblks > 2) {
        return status::unimplemented;
    }

    conf.div = 1;
    int hwd_size = 1;
    int hwd_reduction_size = 1;
    for (int d = 0; d < ndims; d++) {
        conf.src_dims[d] = src_dims[d];
        conf.reduce_dims[d] = conf.dst_dims[d] = dim_t {1};
        conf.is_reduction_dim[d] = conf.src_dims[d] != dst_dims[d];

        if (conf.is_reduction_dim[d]) {
            conf.reduce_dims[d] = conf.src_dims[d];
            conf.div *= conf.reduce_dims[d];
        } else {
            conf.dst_dims[d] = conf.src_dims[d];
        }
        if (d >= 2) {
            hwd_size *= conf.src_dims[d];
            hwd_reduction_size *= conf.reduce_dims[d];
        }
    }
    if (hwd_size != hwd_reduction_size && hwd_reduction_size > 1) {
        return status::unimplemented;
    }

    conf.sub_group_size = 16;
    if (conf.src_dims[1] % conf.sub_group_size != 0) {
        return status_t::dnnl_unimplemented;
    }
    conf.initial_c_chunks
            = std::min(static_cast<int>(conf.src_dims[1]) / conf.sub_group_size,
                    conf.c_block_size / conf.sub_group_size);

    std::tie(conf.initial_n_chunk_size, conf.initial_n_chunks)
            = get_initial_n_split(conf.src_dims[0], conf.is_reduction_dim[0]);

    const auto get_reduction_elems_per_wi = [this]() {
        return conf.initial_n_chunk_size * conf.initial_c_chunks
                * conf.initial_hwd_chunk_size;
    };
    const auto get_wi_per_hwd = [this]() {
        return ceil(static_cast<float>(conf.initial_hwd_dim)
                / conf.initial_hwd_chunk_size);
    };
    const auto get_used_threads_num = [this, get_wi_per_hwd]() {
        return conf.initial_n_chunks * conf.src_dims[1]
                / (conf.sub_group_size * conf.initial_c_chunks)
                * get_wi_per_hwd();
    };

    if (hwd_reduction_size == 1) {
        conf.initial_hwd_chunk_size = 1;
        // If there is no HWD reduction use vectors only to read whole C block
        conf.vector_size = conf.initial_c_chunks;
        conf.initial_hwd_dim = hwd_size;
        conf.final_hwd_dim = hwd_size;
        conf.final_hwd_chunk_size = 1;
    } else {
        // Start with such constant and try to adjust that with heuristics
        conf.initial_hwd_chunk_size = 64;
        if (conf.n_block_size > 1 || conf.src_dims[1] < conf.c_block_size) {
            conf.vector_size = conf.initial_c_chunks;
        } else {
            conf.vector_size = 8;
        }
        conf.initial_hwd_dim = hwd_reduction_size;

        // Experimentally selected values
        constexpr int min_elems_per_wi = 64;
        constexpr int max_wi_per_hwd = 512;
        const int min_threads = num_threads;

        while (get_used_threads_num() < min_threads
                && get_reduction_elems_per_wi() > min_elems_per_wi
                && get_wi_per_hwd() < max_wi_per_hwd) {
            conf.initial_hwd_chunk_size /= 2;
        }
        while ((get_used_threads_num() > min_threads
                       && get_reduction_elems_per_wi() < min_elems_per_wi)
                || get_wi_per_hwd() > max_wi_per_hwd) {
            conf.initial_hwd_chunk_size *= 2;
        }

        while (conf.vector_size > conf.initial_hwd_chunk_size) {
            conf.vector_size /= 2;
        }
        conf.final_hwd_dim = get_wi_per_hwd();
        conf.final_hwd_chunk_size = conf.final_hwd_dim;
    }

    conf.final_c_dim = conf.is_reduction_dim[1]
            ? conf.src_dims[1] / (conf.sub_group_size * conf.initial_c_chunks)
            : conf.src_dims[1];
    conf.final_c_chunk_size = conf.is_reduction_dim[1]
            ? conf.src_dims[1] / (conf.sub_group_size * conf.initial_c_chunks)
            : 1;

    conf.final_n_dim = conf.is_reduction_dim[0] ? conf.initial_n_chunks
                                                : conf.src_dims[0];
    conf.final_n_chunk_size
            = conf.is_reduction_dim[0] ? conf.initial_n_chunks : 1;

    int initial_n_chunks_padded, initial_c_padded;
    if (conf.final_c_chunk_size == 1 && conf.final_n_chunk_size == 1
            && conf.final_hwd_chunk_size == 1) {
        conf.skip_final_phase = true;
        // zero pad N and C in initial phase only when there is no final phase
        const int n_padded = utils::rnd_up(conf.src_dims[0], conf.n_block_size);
        initial_n_chunks_padded = ceil(
                static_cast<float>(n_padded) / conf.initial_n_chunk_size);
        initial_c_padded = utils::rnd_up(conf.src_dims[1], conf.c_block_size);
    } else {
        conf.skip_final_phase = false;
        initial_n_chunks_padded = conf.initial_n_chunks;
        initial_c_padded = conf.src_dims[1];
    }

    conf.dispatch.define_dim("INITIAL_N", 0, initial_n_chunks_padded, 1);
    conf.dispatch.define_dim("INITIAL_C", std::min(ndims - 1, 1),
            initial_c_padded, conf.initial_c_chunks);
    conf.dispatch.define_dim("INITIAL_HWD_CHUNK_ID", std::min(ndims - 1, 2),
            conf.final_hwd_dim, 1);
    CHECK(conf.dispatch.vectorize_dim("INITIAL_C", conf.sub_group_size));
    conf.dispatch.set_kernel_attr_suffix("INITIAL");
    conf.dispatch.generate();

    const int final_n_padded
            = utils::rnd_up(conf.final_n_dim, conf.n_block_size);
    const int final_n_chunks_padded
            = utils::div_up(final_n_padded, conf.final_n_chunk_size);
    conf.finilize_dispatch.define_dim("FINAL_N", 0, final_n_chunks_padded);
    const int final_c_padded
            = utils::rnd_up(conf.final_c_dim, conf.c_block_size);
    const int final_c_chunks_padded
            = utils::div_up(final_c_padded, conf.final_c_chunk_size);
    conf.finilize_dispatch.define_dim(
            "FINAL_C", std::min(ndims - 1, 1), final_c_chunks_padded);
    conf.finilize_dispatch.define_dim("FINAL_HWD", std::min(ndims - 1, 2),
            conf.final_hwd_dim / conf.final_hwd_chunk_size);
    conf.finilize_dispatch.set_kernel_attr_suffix("FINAL");
    conf.finilize_dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const reduction_conf_t &conf) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(conf.src_type);

    kernel_ctx.define_int("INITIAL_N", conf.src_dims[0]);
    kernel_ctx.define_int("INITIAL_C", conf.src_dims[1]);
    kernel_ctx.define_int("INITIAL_C_CHUNKS", conf.initial_c_chunks);
    kernel_ctx.define_int("INITIAL_N_CHUNKS", conf.initial_n_chunks);
    kernel_ctx.define_int("SKIP_FINAL_PHASE", conf.skip_final_phase);
    kernel_ctx.define_int("FINAL_N_DIM", conf.final_n_dim);
    kernel_ctx.define_int("FINAL_N_CHUNK_SIZE", conf.final_n_chunk_size);
    kernel_ctx.define_int("INITIAL_N_CHUNK_SIZE", conf.initial_n_chunk_size);
    kernel_ctx.define_int("FINAL_C_DIM", conf.final_c_dim);
    kernel_ctx.define_int("FINAL_C_CHUNK_SIZE", conf.final_c_chunk_size);
    kernel_ctx.define_int("INITIAL_HWD_DIM", conf.initial_hwd_dim);
    kernel_ctx.define_int("FINAL_HWD_DIM", conf.final_hwd_dim);
    kernel_ctx.define_int(
            "INITIAL_HWD_CHUNK_SIZE", conf.initial_hwd_chunk_size);
    kernel_ctx.define_int("FINAL_HWD_CHUNK_SIZE", conf.final_hwd_chunk_size);
    kernel_ctx.define_int("DST_N", conf.dst_dims[0]);
    kernel_ctx.define_int("DST_C", conf.dst_dims[1]);
    kernel_ctx.define_int(
            "DST_N_PADDED", utils::rnd_up(conf.dst_dims[0], conf.n_block_size));
    kernel_ctx.define_int(
            "DST_C_PADDED", utils::rnd_up(conf.dst_dims[1], conf.c_block_size));

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("C_BLOCK_SIZE", conf.c_block_size);
    kernel_ctx.define_int("N_BLOCK_SIZE", conf.n_block_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vector_size);
    kernel_ctx.define_int("REDUCTION_SIZE", conf.div);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    kernel_ctx.define_int("IS_N_REDUCED", conf.is_reduction_dim[0]);
    kernel_ctx.define_int("IS_C_REDUCED", conf.is_reduction_dim[1]);
    kernel_ctx.define_int(
            "IS_HWD_REDUCED", conf.final_hwd_dim < conf.initial_hwd_dim);

    switch (conf.alg) {
        case reduction_max: kernel_ctx.define_int("IS_MAX", 1); break;
        case reduction_min: kernel_ctx.define_int("IS_MIN", 1); break;
        case reduction_mean: kernel_ctx.define_int("IS_MEAN", 1); break;
        case reduction_sum: kernel_ctx.define_int("IS_SUM", 1); break;
        case reduction_mul: kernel_ctx.define_int("IS_MUL", 1); break;
        case reduction_norm_lp_max:
            kernel_ctx.define_int("IS_LP_MAX", 1);
            break;
        case reduction_norm_lp_sum:
            kernel_ctx.define_int("IS_LP_SUM", 1);
            break;
        case reduction_norm_lp_power_p_max:
            kernel_ctx.define_int("IS_P_MAX", 1);
            break;
        case reduction_norm_lp_power_p_sum:
            kernel_ctx.define_int("IS_P_SUM", 1);
            break;
        default: return status::invalid_arguments;
    }

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);
    def_dispatch(kernel_ctx, conf.finilize_dispatch);

    return status::success;
}

status_t gen9_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

void gen9_reduction_t::pd_t::init_scratchpad() {
    const size_t size = utils::rnd_up(conf.final_n_dim, conf.n_block_size)
            * utils::rnd_up(conf.final_c_dim, conf.c_block_size)
            * conf.final_hwd_dim;

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t gen9_reduction_t::execute_gen9(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    std::unique_ptr<memory_storage_t> temp_reduce
            = ctx.get_scratchpad_grantor().get_memory_storage(
                    memory_tracking::names::key_reduction);
    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t reduction_arg_list;
    reduction_arg_list.set(0, src);
    reduction_arg_list.set(1, conf.skip_final_phase ? dst : *temp_reduce);
    auto initial_nd_range = conf.dispatch.nd_range();
    status_t status = parallel_for(
            ctx, initial_nd_range, initial_kernel, reduction_arg_list);

    if (!conf.skip_final_phase) {
        if (status != status::success) return status;
        compute::kernel_arg_list_t final_reduction_arg_list;
        final_reduction_arg_list.set(0, *temp_reduce);
        final_reduction_arg_list.set(1, dst);
        auto final_nd_range = conf.finilize_dispatch.nd_range();
        return parallel_for(
                ctx, final_nd_range, final_kernel, final_reduction_arg_list);
    } else {
        return status;
    }
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
