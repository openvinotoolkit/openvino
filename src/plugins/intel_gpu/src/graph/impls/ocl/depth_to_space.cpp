// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "depth_to_space/depth_to_space_kernel_selector.h"
#include "depth_to_space/depth_to_space_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "common_types.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct depth_to_space_impl : typed_primitive_impl_ocl<depth_to_space> {
    using parent = typed_primitive_impl_ocl<depth_to_space>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<depth_to_space_impl>(*this);
    }

public:
    static primitive_impl* create(const depth_to_space_node& arg) {
        const auto& prim = arg.get_primitive();

        const auto& param_info = kernel_impl_params(arg.get_program(), prim, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto depth_to_space_params = get_default_params<kernel_selector::depth_to_space_params>(param_info);
        auto depth_to_space_optional_params =
            get_default_optional_params<kernel_selector::depth_to_space_optional_params>(arg.get_program());

        depth_to_space_params.block_size = prim->block_size;
        depth_to_space_params.mode = prim->mode == depth_to_space_mode::blocks_first ? kernel_selector::depth_to_space_mode::BLOCKS_FIRST
                                                                                                    : kernel_selector::depth_to_space_mode::DEPTH_FIRST;

        auto& kernel_selector = kernel_selector::depth_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(depth_to_space_params, depth_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto depth_to_space = new depth_to_space_impl(arg, best_kernels[0]);

        return depth_to_space;
    }
};

namespace detail {

attach_depth_to_space_impl::attach_depth_to_space_impl() {
    implementation_map<depth_to_space>::add(impl_types::ocl, depth_to_space_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
