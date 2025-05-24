/*******************************************************************************
* Copyright (c) 2022-2025 Intel Corporation
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

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, mem_layout layout_a, mem_layout layout_b,
        mem_layout layout_c, mem_space mem_space_a, mem_space mem_space_b,
        mem_space mem_space_c, uint32_t local_kslicing,
        uint32_t global_kslicing, mma_engine engine,
        uint32_t periodic_sync_interval, uint32_t prefetch_distance,
        gpu_arch arch_tag, uint32_t snake_swizzle = 0, bool unaligned = false>
struct gemm_universal {
    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;

    using mem_desc_a
            = mem_desc_t<dtype_a, layout_a, mem_space_a, unaligned ? 1 : 8>;
    using mem_desc_b
            = mem_desc_t<dtype_b, layout_b, mem_space_b, unaligned ? 1 : 8>;
    using mem_desc_c
            = mem_desc_t<dtype_c, layout_c, mem_space_c, unaligned ? 1 : 8>;

    using compute_attr = typename std::conditional<engine == mma_engine::fpu,
            compute_attr_t<dtype_acc, dtype_acc, dtype_acc>,
            compute_attr_t<dtype_a, dtype_b, dtype_acc>>::type;

    using perf_tuning_knob = perf_tuning_knob_t<sg_k, prefetch_distance,
            periodic_sync_interval>;

    using compute_policy_0 =
            typename std::conditional<engine == mma_engine::fpu,
                    compute_policy_default_fpu<compute_attr, perf_tuning_knob,
                            arch_tag>,
                    compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                            arch_tag>>::type;
    using compute_policy = typename std::conditional<unaligned,
            compute_policy_unaligned_xmx<compute_attr, perf_tuning_knob,
                    arch_tag>,
            compute_policy_0>::type;
    using pre_processing = pre_processing_default_t<tile_shape, arch_tag>;
    using gemm = gemm_t<compute_policy, tile_shape, mem_desc_a, mem_desc_b,
            pre_processing>;

    LORA_POST_OP_DEFINITIONS

    using tile_op_t = subgroup::chained_tile_op_t<LORA_POST_OP_LIST>;
    using epilogue = epilogue_t<
            epilogue_policy_tile_op<tile_op_t, arch_tag,
                    unaligned ? msg_type::unaligned_2d : msg_type::block_2d>,
            tile_shape, mem_desc_c>;

    using epilogue_args_t = typename epilogue::arguments_t;

    using group_swizzle_t = kernel::group_swizzle_default<arch_tag>;

    using gemm_op_t = kernel::gemm_universal_t<
            kernel::dispatch_policy_kslicing<group_swizzle_t, global_kslicing,
                    local_kslicing>,
            gemm, epilogue>;

    static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
    static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();

    inline static void run(sycl::nd_item<3> &item, dtype_a *a, dtype_b *b,
            typename epilogue::mem_desc_c_t::base_t c, dtype_acc *acc,
            uint32_t *cnt, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k,
            uint32_t lda, uint32_t ldb, uint32_t ldc LORA_POST_OP_ARGS) {
        gemm_op_t gemm_op;

        LORA_POST_OP_SHAPE_DEFINITIONS
        epilogue_args_t epilogue_args;
        epilogue_args.init({LORA_POST_OP_EPILOGUE_INIT_ARGS});

        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, a, lda, b, ldb,
                c.base, ldc, acc, cnt, epilogue_args);
        gemm_op(item, arg);
    }
};
