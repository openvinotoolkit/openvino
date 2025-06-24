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

template <typename dtype_in, typename dtype_out, typename dtype_acc,
        mem_layout mem_layout_in, mem_layout mem_layout_out, uint32_t wg_m,
        uint32_t wg_n, uint32_t sg_m, uint32_t sg_n>
struct postop {
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;

    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;

    using tile_desc_t
            = subgroup::tile_desc_t<sg_n, sg_m, sg_n, sg_m, reg_layout::tiled>;

    using tile_in_t = subgroup::tile_t<dtype_in, tile_desc_t>;
    using tile_acc_t = subgroup::tile_t<dtype_acc, tile_desc_t>;
    using tile_out_t = subgroup::tile_t<dtype_out, tile_desc_t>;

    using mem_desc_in_t
            = mem_desc_t<dtype_in, mem_layout_in, mem_space::global, 1>;
    using mem_desc_out_t
            = mem_desc_t<dtype_out, mem_layout_out, mem_space::global, 1>;

    using payload_input_t = subgroup::mem_payload_t<mem_desc_in_t, tile_desc_t,
            msg_type::unaligned_2d, arch_tag>;

    XETLA_POST_OP_DEFINITIONS

    using tile_op_t = subgroup::chained_tile_op_t<XETLA_POST_OP_LIST>;
    using epilogue = epilogue_t<epilogue_policy_tile_op<tile_op_t, arch_tag,
                                        msg_type::unaligned_2d>,
            tile_shape, mem_desc_out_t>;
    using epilogue_args_t = typename epilogue::arguments_t;

    inline static void run(sycl::nd_item<3> &item, uint32_t mat_m,
            uint32_t mat_n, dtype_in *in, dtype_out *out XETLA_POST_OP_ARGS) {
        uint32_t ldc = mat_n;
        int32_t start_m = item.get_group(1) * wg_m;
        int32_t start_n = item.get_group(2) * wg_n;

        uint32_t boundary_m
                = (start_m + wg_m) > mat_m ? mat_m : (start_m + wg_m);
        uint32_t boundary_n
                = (start_n + wg_n) > mat_n ? mat_n : (start_n + wg_n);

        typename tile_shape::work_group_t g;
        g.init(item.get_local_linear_id());

        mem_desc_in_t mem_desc_in;
        mem_desc_out_t mem_desc_out;

        int sg_start_n = start_n + (g.get_id() % tile_shape::wg_size_x) * sg_n;
        int sg_start_m = start_m + (g.get_id() / tile_shape::wg_size_x) * sg_m;

        mem_desc_in.init(
                in, {boundary_n, boundary_m, ldc}, {sg_start_n, sg_start_m});
        mem_desc_out.init(
                out, {boundary_n, boundary_m, ldc}, {start_n, start_m});

        tile_in_t mat_in;
        payload_input_t input_payload(mem_desc_in);

        subgroup::tile_load<cache_hint::uncached, cache_hint::uncached>(
                mat_in, input_payload);

        tile_acc_t mat_acc;
        subgroup::elemwise_cvt(mat_acc, mat_in);

        XETLA_POST_OP_SHAPE_DEFINITIONS
        epilogue_args_t epilogue_args;
        epilogue_args.init({XETLA_POST_OP_EPILOGUE_INIT_ARGS});

        epilogue epilogue;
        epilogue(g, mat_acc, mem_desc_out, epilogue_args, 0, 0);
    }
};
